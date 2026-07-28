"""
Enterprise scenario: two initiatives under one governance layer.

This is the demo that answers the executive question — "I want to fund
the strategic initiative heavily and keep routine internal work cheap,
but I cannot personally review what thousands of engineers are doing."

Two initiatives share the same engine and the same taxonomy, but carry
different budget caps:

  agentic_commerce  — the strategic bet. Generous cap.
  internal_tools    — routine internal work. Tight cap.

Nothing about the initiatives is hardcoded into the engine. Each request
simply carries a `tags` dict, and the governance layer attributes cost
against those tags. An executive sets policy per initiative; the system
enforces it on every request; only exceptions surface.

Run with:  python -m examples.enterprise_demo
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    AgentOrchestrator,
    CostGovernor,
    DecisionLogger,
    EmbeddingClassifier,
    EnsembleClassifier,
    ExportBuilder,
    GovernanceFeed,
    RuleBasedClassifier,
    TagBudget,
    TokenTracker,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "enterprise_requests.json")
EXPORT_PATH = os.path.join(ROOT, "dashboard", "enterprise_export.json")

# The executive decision, expressed as two numbers.
#
# agentic_commerce is the strategic bet: funded generously, expected to
# reach for expensive models, and left alone as long as it stays inside
# a large envelope. internal_tools is routine work: held to a tight cap,
# because routine work reaching for expensive models is precisely the
# drift worth catching.
#
# Caps are scaled to this demo's synthetic volume (tens of requests, not
# millions) so the mechanism is visible in a single run. Real caps would
# be monthly figures from finance.
TAG_BUDGETS = [
    TagBudget("initiative", "agentic_commerce", monthly_cap_usd=0.08),
    TagBudget("initiative", "internal_tools", monthly_cap_usd=0.008),
]


def main() -> None:
    with open(DATA_PATH) as f:
        requests = json.load(f)

    classifier = EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()])
    orchestrator = AgentOrchestrator(classifier=classifier)
    tracker = TokenTracker()
    logger = DecisionLogger()
    governor = CostGovernor(tag_budgets=TAG_BUDGETS)

    for i, item in enumerate(requests):
        task = orchestrator.process(
            task_id=f"E{i:03d}", raw_text=item["text"], tags=item.get("tags", {})
        )
        tracker.record(task)
        logger.record(task)

    print(f"Processed {len(requests)} requests across 2 initiatives.\n")

    print("=== Portfolio rollup (the exec view) ===\n")
    header = f"{'initiative':<20} {'tasks':<7} {'spend($)':<11} {'esc rate':<10} {'model mix'}"
    print(header)
    print("-" * (len(header) + 20))
    for t in tracker.tag_summaries(tag_key="initiative"):
        mix = ", ".join(f"{k}={v}" for k, v in sorted(t.models_used.items()))
        print(
            f"{t.tag_value:<20} {t.task_count:<7} {t.total_cost_usd:<11.4f} "
            f"{t.escalation_rate:<10.0%} {mix}"
        )

    print("\n=== Budget exceptions (what actually reaches the exec) ===\n")
    alerts = governor.check_tag_budgets(tracker)
    breached = [a for a in alerts if a.is_over_cap]
    for a in alerts:
        flag = "OVER CAP" if a.is_over_cap else "ok"
        print(
            f"{a.tag_key}:{a.tag_value:<22} ${a.spend_usd:<9.4f} / "
            f"${a.cap_usd:<8} cap  ({a.pct_of_cap:.0%})  [{flag}]"
        )
    print(
        f"\n{len(breached)} of {len(alerts)} initiatives breached cap — "
        "only these would page a human."
    )

    print("\n=== Per-initiative routing health ===\n")
    for initiative in ("agentic_commerce", "internal_tools"):
        h = logger.health_snapshot(initiative=initiative)
        print(
            f"{initiative:<20} decisions={h['decisions']:<4} "
            f"escalation={h['escalation_rate']:.0%}  "
            f"manual_fallback={h['manual_fallback_rate']:.0%}  "
            f"multi_intent={h['multi_intent_rate']:.0%}"
        )

    exporter = ExportBuilder(tracker, governor, logger)
    exporter.write(EXPORT_PATH)

    # Prove the contract round-trips: read it back through the public
    # reader exactly as a separate dashboard project would.
    feed = GovernanceFeed.from_file(EXPORT_PATH)
    print(f"\n=== Contract round-trip (schema {feed.schema_version}) ===\n")
    print(f"initiatives in feed : {len(feed.by_tag('initiative'))}")
    print(f"decisions in feed   : {len(feed.decisions())}")
    print(f"agentic_commerce    : {len(feed.decisions(initiative='agentic_commerce'))} decisions")
    print(f"over-cap alerts     : {len(feed.alerts(over_cap_only=True))}")
    print(f"\nExport written to {EXPORT_PATH}")
    print("A separate dashboard project consumes exactly this file — nothing else.")


if __name__ == "__main__":
    main()
