"""
End-to-end demo: classify a batch of sample requests, route them through
the agent orchestrator, track token/cost usage per task type, log every
routing decision, check budgets, and emit the versioned export document.

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
    DecisionLogger,
    EmbeddingClassifier,
    EnsembleClassifier,
    ExportBuilder,
    RuleBasedClassifier,
    TaskType,
    TokenTracker,
    escalation_sensitivity,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "sample_requests.json")
DASHBOARD_DATA_PATH = os.path.join(ROOT, "dashboard", "data.json")
EXPORT_PATH = os.path.join(ROOT, "dashboard", "governance_export.json")
DECISION_LOG_PATH = os.path.join(ROOT, "dashboard", "decisions.jsonl")


def main() -> None:
    with open(DATA_PATH) as f:
        requests = json.load(f)

    # An ensemble of the cheap deterministic classifier and the semantic
    # one: agreement raises confidence, disagreement lowers it, and the
    # orchestrator reacts to that confidence by escalating model tier.
    classifier = EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()])

    orchestrator = AgentOrchestrator(classifier=classifier)
    tracker = TokenTracker()
    logger = DecisionLogger()

    governor = CostGovernor(budgets=[
        Budget(TaskType.SUPPORT_QUERY, monthly_cap_usd=5.00),
        Budget(TaskType.CODE_REVIEW, monthly_cap_usd=5.00),
        Budget(TaskType.DATA_LOOKUP, monthly_cap_usd=3.00),
        Budget(TaskType.REPORT_GENERATION, monthly_cap_usd=3.00),
        Budget(TaskType.CONTENT_DRAFTING, monthly_cap_usd=3.00),
        Budget(TaskType.WORKFLOW_AUTOMATION, monthly_cap_usd=3.00),
    ])

    print(f"Processing {len(requests)} incoming requests...\n")
    header = (
        f"{'task_id':<8} {'task_type':<20} {'conf.':<6} {'mode':<18} "
        f"{'model':<16} {'esc':<5} {'cost($)':<8}"
    )
    print(header)
    print("-" * len(header))

    for i, raw_text in enumerate(requests):
        task = orchestrator.process(task_id=f"T{i:03d}", raw_text=raw_text)
        cost = tracker.record(task)
        logger.record(task)
        model = task.metadata.get("routed_model", "-")
        esc = "yes" if task.metadata.get("escalated") else "-"
        print(
            f"{task.task_id:<8} {task.task_type.value:<20} {task.classification_confidence:<6} "
            f"{task.execution_mode.value:<18} {model:<16} {esc:<5} {cost:<8.4f}"
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
    print(
        "Total estimated avoided cost "
        f"(PROJECTION vs. ungoverned baseline): ${total_avoided:.4f}"
    )

    print("\n=== Routing health ===\n")
    health = logger.health_snapshot()
    print(f"decisions logged     : {health['decisions']}")
    print(f"escalation rate      : {health['escalation_rate']:.0%}")
    print(f"manual fallback rate : {health['manual_fallback_rate']:.0%}")
    print(f"multi-intent rate    : {health['multi_intent_rate']:.0%}")
    print(f"model mix            : {health['model_mix']}")

    print("\n=== Escalation threshold calibration ===\n")
    print("Where the cost cliffs are for this traffic. Pick a threshold in a")
    print("gap between clusters, not on top of one.\n")
    print(f"{'threshold':<11} {'manual':<9} {'escalated':<11} {'baseline':<9}")
    print("-" * 42)
    for row in escalation_sensitivity(classifier, requests, thresholds=[0.65, 0.7, 0.75, 0.8, 0.85, 0.9]):
        marker = "  <- current" if row["threshold"] == orchestrator.config.escalation_confidence_threshold else ""
        print(
            f"{row['threshold']:<11} {row['manual_rate']:<9.0%} "
            f"{row['escalation_rate']:<11.0%} {row['baseline_rate']:<9.0%}{marker}"
        )

    print("\n=== Budget check ===\n")
    for a in governor.check_budgets(tracker):
        flag = "OVER CAP" if a.is_over_cap else "ok"
        print(f"{a.task_type.value:<20} ${a.spend_usd:<8} / ${a.cap_usd:<8} cap  [{flag}]")

    # Emit the versioned contract + the raw decision log.
    exporter = ExportBuilder(tracker, governor, logger)
    exporter.write(EXPORT_PATH)
    logger.write_jsonl(DECISION_LOG_PATH)

    # Backwards-compatible payload for the bundled demo dashboard.
    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump({
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
        }, f, indent=2)

    print(f"\nExport contract written to {EXPORT_PATH}")
    print(f"Decision log written to    {DECISION_LOG_PATH}")
    print(f"Dashboard data written to  {DASHBOARD_DATA_PATH}")
    print("Serve the dashboard/ folder over http:// to view it.")


if __name__ == "__main__":
    main()
