"""
End-to-end demo: classify a batch of sample requests, route them through
the agent orchestrator, track token/cost usage per task type, check
budgets, and estimate avoided cost versus an ungoverned baseline.

Run with:  python -m examples.demo
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    AgentOrchestrator,
    Budget,
    CostGovernor,
    TaskType,
    TokenTracker,
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_requests.json"
)
DASHBOARD_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "data.json"
)


def main() -> None:
    with open(DATA_PATH) as f:
        requests = json.load(f)

    orchestrator = AgentOrchestrator()
    tracker = TokenTracker()

    governor = CostGovernor(budgets=[
        Budget(TaskType.SUPPORT_QUERY, monthly_cap_usd=5.00),
        Budget(TaskType.CODE_REVIEW, monthly_cap_usd=5.00),
        Budget(TaskType.DATA_LOOKUP, monthly_cap_usd=3.00),
        Budget(TaskType.REPORT_GENERATION, monthly_cap_usd=3.00),
        Budget(TaskType.CONTENT_DRAFTING, monthly_cap_usd=3.00),
        Budget(TaskType.WORKFLOW_AUTOMATION, monthly_cap_usd=3.00),
    ])

    print(f"Processing {len(requests)} incoming requests...\n")
    print(f"{'task_id':<8} {'task_type':<20} {'conf.':<6} {'mode':<18} {'model':<16} {'cost($)':<8}")
    print("-" * 82)

    for i, raw_text in enumerate(requests):
        task = orchestrator.process(task_id=f"T{i:03d}", raw_text=raw_text)
        cost = tracker.record(task)
        model = task.metadata.get("routed_model", "-")
        print(
            f"{task.task_id:<8} {task.task_type.value:<20} {task.classification_confidence:<6} "
            f"{task.execution_mode.value:<18} {model:<16} {cost:<8.4f}"
        )

    print("\n=== Usage summary by task type ===\n")
    summaries = tracker.summaries()
    for s in summaries:
        avoided = CostGovernor.estimate_avoided_cost(s)
        print(
            f"{s.task_type.value:<20} tasks={s.task_count:<4} "
            f"cost=${s.total_cost_usd:.4f}  est. avoided=${avoided:.4f}"
        )

    total_cost = tracker.total_cost_usd()
    total_avoided = sum(CostGovernor.estimate_avoided_cost(s) for s in summaries)
    print(f"\nTotal governed cost: ${total_cost:.4f}")
    print(f"Total estimated avoided cost (illustrative, vs. ungoverned baseline): ${total_avoided:.4f}")

    print("\n=== Budget check ===\n")
    alerts = governor.check_budgets(tracker)
    for a in alerts:
        flag = "OVER CAP" if a.is_over_cap else "ok"
        print(f"{a.task_type.value:<20} ${a.spend_usd:<8} / ${a.cap_usd:<8} cap  [{flag}]")

    # Export data for the dashboard visualization
    dashboard_payload = {
        "summaries": [
            {
                "task_type": s.task_type.value,
                "task_count": s.task_count,
                "total_cost_usd": round(s.total_cost_usd, 4),
                "avg_cost_per_task": round(s.avg_cost_per_task, 4),
                "estimated_avoided_cost_usd": CostGovernor.estimate_avoided_cost(s),
            }
            for s in summaries
        ],
        "total_cost_usd": round(total_cost, 4),
        "total_estimated_avoided_cost_usd": round(total_avoided, 4),
    }
    os.makedirs(os.path.dirname(DASHBOARD_DATA_PATH), exist_ok=True)
    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump(dashboard_payload, f, indent=2)
    print(f"\nDashboard data written to {DASHBOARD_DATA_PATH}")
    print("Open dashboard/index.html in a browser to view it.")


if __name__ == "__main__":
    main()
