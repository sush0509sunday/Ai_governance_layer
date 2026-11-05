"""
Route synthetic Jira work items and print the view finance would see.

Each request carries its work item. The issue type (or a label) sets the
task type and baseline model tier, the component sets risk, and the prompt
classifier only refines. Every field becomes a cost-attribution tag.

All work items, prompts and prices are synthetic. Run from the repo root:

    python -m examples.jira_routing_demo
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    AgentOrchestrator,
    DecisionLogger,
    EmbeddingClassifier,
    EnsembleClassifier,
    RuleBasedClassifier,
    TokenTracker,
    WorkItem,
    WorkItemRouter,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "jira_work_items.json")


def main() -> None:
    with open(DATA_PATH) as f:
        rows = json.load(f)

    orchestrator = AgentOrchestrator(
        classifier=EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()])
    )
    router = WorkItemRouter(orchestrator)
    tracker, logger = TokenTracker(), DecisionLogger()

    print(f"{'work item':<14} {'issue type':<16} {'component':<10} {'task type':<17} {'model':<17} note")
    print("-" * 96)
    for row in rows:
        item = WorkItem(**row["work_item"])
        task = router.route(item, row["prompt"])
        tracker.record(task)
        logger.record(task)
        notes = []
        if task.metadata.get("human_review"):
            notes.append("high-risk: human review")
        if task.metadata.get("signal_conflict"):
            notes.append(f"prompt says {task.metadata['prompt_task_type']}: escalated")
        elif task.metadata.get("escalated"):
            notes.append("low prompt confidence: escalated")
        if task.metadata.get("routing_signal") == "prompt_only":
            notes.append("no rule: prompt-only")
        print(
            f"{item.key:<14} {item.issue_type:<16} {item.component or '-':<10} "
            f"{task.task_type.value:<17} {task.metadata.get('routed_model', '-'):<17} {'; '.join(notes)}"
        )

    print("\nWhat finance sees (synthetic prices)\n")
    for key in ("project", "issue_type", "component"):
        print(f"Spend by {key}:")
        for s in tracker.tag_summaries(key):
            print(f"  {s.tag_value:<18} {s.task_count:>3} requests   ${s.total_cost_usd:.4f}   escalated {s.escalation_rate:.0%}")
        print()

    issues = tracker.tag_summaries("work_item")
    total = sum(s.total_cost_usd for s in issues)
    print(f"Average AI cost per work item: ${total / len(issues):.4f} across {len(issues)} items")

    print("\nGovernance health:")
    print(f"  human review (high-risk components): {logger.human_review_rate():.0%}")
    print(f"  escalation rate:                     {logger.escalation_rate():.0%}")
    print(f"  manual fallback rate:                {logger.manual_fallback_rate():.0%}")

    print("\nLineage for one work item:")
    record = next(r for r in logger if r.task_id == "CHECKOUT-211")
    print(f"  {record.task_id} -> {record.task_type} -> {record.routed_model} "
          f"-> {record.input_tokens + record.output_tokens} tokens -> ${record.cost_usd:.4f}"
          f" (human review: {record.human_review})")

    print("\nEscalation rate by requester (a simple guard against gaming):")
    requesters = sorted({s.tag_value for s in tracker.tag_summaries("requester")})
    for name in requesters:
        print(f"  {name:<8} {logger.escalation_rate(requester=name):.0%}")


if __name__ == "__main__":
    main()
