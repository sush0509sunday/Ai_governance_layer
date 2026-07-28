"""
Decision logging and routing observability.

Routing write-ups almost universally end with "and of course you should
log every routing decision" — and then don't. This module is that step,
implemented.

Every task that passes through the governance layer produces one
`DecisionRecord`: what it was classified as, how confident the
classifier was, which method produced that classification, whether the
request was escalated to a pricier model tier or handed to a human,
which model actually ran it, and what it cost. That record is the audit
trail behind every dollar in the cost report.

The derived rates matter as much as the raw log:

- A rising **escalation rate** means the classifier is getting less
  confident about incoming traffic — usually the first signal that real
  usage has drifted away from the taxonomy the classifier was built for.
- A rising **manual fallback rate** means more work is bypassing
  automation entirely, which erodes the ROI case.
- A rising **overlap rate** means requests increasingly span multiple
  categories, which is the signal to split or redesign the taxonomy.

None of these are visible from a cost total alone. They are what turns
"our AI bill went up" into "our AI bill went up *because* classification
quality is degrading."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .models import ExecutionMode, GovernedTask, PriceTable

SCHEMA_VERSION = "1.0"


@dataclass
class DecisionRecord:
    """One immutable row describing a single routing decision."""

    task_id: str
    timestamp: str
    task_type: str
    classification_confidence: float
    classification_method: str
    execution_mode: str
    routed_model: Optional[str]
    baseline_model: Optional[str]
    escalated: bool
    multi_intent: bool
    reason: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionLogger:
    """Collects `DecisionRecord`s and computes routing health metrics."""

    def __init__(self, price_table: PriceTable | None = None):
        self.price_table = price_table or PriceTable()
        self._records: list[DecisionRecord] = []

    def record(self, task: GovernedTask) -> DecisionRecord:
        usage = task.token_usage
        cost = usage.cost_usd(self.price_table) if usage else 0.0

        record = DecisionRecord(
            task_id=task.task_id,
            timestamp=task.created_at.astimezone(timezone.utc).isoformat(),
            task_type=task.task_type.value,
            classification_confidence=task.classification_confidence,
            classification_method=task.metadata.get("classification_method", "unknown"),
            execution_mode=task.execution_mode.value,
            routed_model=task.metadata.get("routed_model"),
            baseline_model=task.metadata.get("baseline_model"),
            escalated=bool(task.metadata.get("escalated", False)),
            multi_intent=bool(task.metadata.get("multi_intent", False)),
            reason=task.metadata.get("reason"),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cost_usd=round(cost, 6),
            tags=dict(task.tags),
        )
        self._records.append(record)
        return record

    # ---- access -----------------------------------------------------

    def records(self, **tag_filters: str) -> list[DecisionRecord]:
        """All records, optionally filtered to those matching every tag given."""
        if not tag_filters:
            return list(self._records)
        return [
            r for r in self._records
            if all(r.tags.get(k) == v for k, v in tag_filters.items())
        ]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[DecisionRecord]:
        return iter(self._records)

    # ---- derived routing-health metrics -----------------------------

    def _rate(self, predicate, **tag_filters: str) -> float:
        records = self.records(**tag_filters)
        if not records:
            return 0.0
        return round(sum(1 for r in records if predicate(r)) / len(records), 4)

    def escalation_rate(self, **tag_filters: str) -> float:
        """Share of governed tasks bumped to a pricier model tier."""
        return self._rate(lambda r: r.escalated, **tag_filters)

    def manual_fallback_rate(self, **tag_filters: str) -> float:
        """Share of tasks the system declined to automate."""
        return self._rate(
            lambda r: r.execution_mode == ExecutionMode.UNGOVERNED_MANUAL.value,
            **tag_filters,
        )

    def multi_intent_rate(self, **tag_filters: str) -> float:
        """Share of requests whose top two candidate categories were near-tied."""
        return self._rate(lambda r: r.multi_intent, **tag_filters)

    def model_mix(self, **tag_filters: str) -> dict[str, int]:
        """How many tasks ran on each model tier."""
        mix: dict[str, int] = {}
        for r in self.records(**tag_filters):
            if r.routed_model:
                mix[r.routed_model] = mix.get(r.routed_model, 0) + 1
        return dict(sorted(mix.items(), key=lambda kv: kv[1], reverse=True))

    def health_snapshot(self, **tag_filters: str) -> dict:
        """The four numbers worth putting in front of a human."""
        return {
            "decisions": len(self.records(**tag_filters)),
            "escalation_rate": self.escalation_rate(**tag_filters),
            "manual_fallback_rate": self.manual_fallback_rate(**tag_filters),
            "multi_intent_rate": self.multi_intent_rate(**tag_filters),
            "model_mix": self.model_mix(**tag_filters),
        }

    # ---- persistence ------------------------------------------------

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(r.to_dict()) for r in self._records)

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl() + ("\n" if self._records else ""))
        return path

    @classmethod
    def load_jsonl(cls, path: str | Path) -> list[DecisionRecord]:
        """Read decision records back — the drill-down path for consumers."""
        lines = Path(path).read_text().splitlines()
        return [DecisionRecord(**json.loads(line)) for line in lines if line.strip()]
