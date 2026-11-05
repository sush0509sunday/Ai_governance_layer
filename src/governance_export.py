"""
The export contract — the stable boundary between this engine and
anything that consumes it.

This module exists because the governance layer is not the end product.
Downstream consumers (an enterprise dashboard, a FinOps tool, a
chargeback system, a weekly exec digest) need this data, and they should
NOT reach into `TokenTracker` or `DecisionLogger` internals to get it.
Once two projects share Python internals, the engine can no longer
change without breaking its consumers.

So the engine emits one versioned document instead, at three grains:

  decisions[]  — one row per routing decision. The drill-down grain.
                 Answers "why did this specific request cost that?"
  summaries[]  — cost rolled up by task type and by tag. The headline
                 grain. Answers "where is the money going?"
  alerts[]     — budget exceptions only. The exec grain. Answers "what
                 needs my attention?" and nothing else.

Three design rules this module enforces, each learned from the way
integrations usually rot:

1. **Aggregations are computed here, never by the consumer.** If both
   the engine and a dashboard can independently compute "total spend for
   code_review", they will eventually disagree, and then no one trusts
   either number. The engine is the single source of truth; consumers
   render what they're given.

2. **The payload is versioned.** `schema_version` ships in every
   document so a consumer can detect an incompatible engine rather than
   silently mis-parsing it. Costs one field now; avoids a painful
   migration later.

3. **Projections are labeled in the data itself, not just the docs.**
   Estimated avoided cost travels with an `is_projection: true` flag and
   its methodology. A number that can be mistaken for a measured result
   eventually will be, and that is a credibility risk rather than a
   formatting nitpick.

Transport is deliberately layered: the same document can be written to a
file, returned from an in-process call, or served over HTTP. Consumers
parse one schema regardless of how they receive it, so a dashboard can
start file-based and move to a service later without changing its
parsing code.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .cost_governor import CostGovernor
from .observability import DecisionLogger
from .token_tracker import TokenTracker

# Bump the minor version for additive changes (new optional fields), the
# major version for anything that would break an existing consumer.
SCHEMA_VERSION = "1.0"

AVOIDED_COST_METHODOLOGY = (
    "Modeled as (governed_cost * ungoverned_multiplier) - governed_cost. The "
    "multiplier is an illustrative placeholder representing costs a governance "
    "layer typically prevents (redundant context, no caching, no model-tier "
    "routing, unmanaged agent loops). This is a PROJECTION, not a measured "
    "result. Replace with a before/after measurement against real historical "
    "spend before reporting it as realized savings."
)


class ExportBuilder:
    """Builds the versioned governance document consumers read."""

    def __init__(
        self,
        tracker: TokenTracker,
        governor: CostGovernor,
        logger: Optional[DecisionLogger] = None,
        ungoverned_multiplier: float = 2.4,
    ):
        self.tracker = tracker
        self.governor = governor
        self.logger = logger
        self.ungoverned_multiplier = ungoverned_multiplier

    def build(self, include_decisions: bool = True) -> dict[str, Any]:
        summaries = self.tracker.summaries()

        total_cost = self.tracker.total_cost_usd()
        total_avoided = sum(
            CostGovernor.estimate_avoided_cost(s, self.ungoverned_multiplier)
            for s in summaries
        )

        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "totals": {
                "total_governed_cost_usd": round(total_cost, 4),
                "estimated_avoided_cost_usd": round(total_avoided, 4),
                "is_projection": True,
                "projection_methodology": AVOIDED_COST_METHODOLOGY,
                "ungoverned_multiplier": self.ungoverned_multiplier,
            },
            "summaries": {
                "by_task_type": [
                    {
                        "task_type": s.task_type.value,
                        "task_count": s.task_count,
                        "total_input_tokens": s.total_input_tokens,
                        "total_output_tokens": s.total_output_tokens,
                        "total_cost_usd": round(s.total_cost_usd, 4),
                        "avg_cost_per_task_usd": round(s.avg_cost_per_task, 6),
                        "estimated_avoided_cost_usd": CostGovernor.estimate_avoided_cost(
                            s, self.ungoverned_multiplier
                        ),
                        "is_projection": True,
                    }
                    for s in summaries
                ],
                "by_tag": [
                    {
                        "tag_key": t.tag_key,
                        "tag_value": t.tag_value,
                        "task_count": t.task_count,
                        "total_cost_usd": round(t.total_cost_usd, 4),
                        "avg_cost_per_task_usd": round(t.avg_cost_per_task, 6),
                        "escalated_count": t.escalated_count,
                        "escalation_rate": round(t.escalation_rate, 4),
                        "models_used": t.models_used,
                    }
                    for t in self.tracker.tag_summaries()
                ],
            },
            "alerts": self._build_alerts(),
        }

        if self.logger is not None:
            document["routing_health"] = self.logger.health_snapshot()
            if include_decisions:
                document["decisions"] = [r.to_dict() for r in self.logger.records()]

        return document

    def _build_alerts(self) -> list[dict[str, Any]]:
        """Exceptions only — this is what an exec-facing consumer reads first."""
        alerts: list[dict[str, Any]] = []

        for a in self.governor.check_budgets(self.tracker):
            alerts.append({
                "scope": "task_type",
                "subject": a.task_type.value,
                "spend_usd": a.spend_usd,
                "cap_usd": a.cap_usd,
                "pct_of_cap": a.pct_of_cap,
                "status": "over_cap" if a.is_over_cap else "ok",
            })

        for a in self.governor.check_tag_budgets(self.tracker):
            alerts.append({
                "scope": "tag",
                "subject": f"{a.tag_key}:{a.tag_value}",
                "tag_key": a.tag_key,
                "tag_value": a.tag_value,
                "spend_usd": a.spend_usd,
                "cap_usd": a.cap_usd,
                "pct_of_cap": a.pct_of_cap,
                "status": "over_cap" if a.is_over_cap else "ok",
            })

        return sorted(alerts, key=lambda a: a["pct_of_cap"], reverse=True)

    def write(self, path: str | Path, include_decisions: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(include_decisions), indent=2))
        return path


class GovernanceFeed:
    """Read-side of the contract — what a consumer imports.

    A downstream project should depend on this class and nothing else in
    this package. It parses the versioned document and exposes typed
    accessors, so a dashboard never needs to know how `TokenTracker` or
    `DecisionLogger` are implemented — or that they exist.
    """

    def __init__(self, document: dict[str, Any]):
        version = document.get("schema_version")
        if version is None:
            raise ValueError("Document has no schema_version — not a governance export.")
        major = version.split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"Incompatible schema version {version!r}; this reader supports "
                f"{SCHEMA_VERSION.split('.')[0]}.x"
            )
        self.document = document

    @classmethod
    def from_file(cls, path: str | Path) -> "GovernanceFeed":
        return cls(json.loads(Path(path).read_text()))

    @property
    def schema_version(self) -> str:
        return self.document["schema_version"]

    @property
    def generated_at(self) -> str:
        return self.document["generated_at"]

    def totals(self) -> dict[str, Any]:
        return self.document["totals"]

    def by_task_type(self) -> list[dict[str, Any]]:
        return self.document["summaries"]["by_task_type"]

    def by_tag(self, tag_key: str | None = None) -> list[dict[str, Any]]:
        rows = self.document["summaries"]["by_tag"]
        if tag_key is None:
            return rows
        return [r for r in rows if r["tag_key"] == tag_key]

    def alerts(self, over_cap_only: bool = False) -> list[dict[str, Any]]:
        rows = self.document.get("alerts", [])
        if over_cap_only:
            return [r for r in rows if r["status"] == "over_cap"]
        return rows

    def routing_health(self) -> dict[str, Any]:
        return self.document.get("routing_health", {})

    def decisions(self, **tag_filters: str) -> list[dict[str, Any]]:
        rows = self.document.get("decisions", [])
        if not tag_filters:
            return rows
        return [
            r for r in rows
            if all(r.get("tags", {}).get(k) == v for k, v in tag_filters.items())
        ]
