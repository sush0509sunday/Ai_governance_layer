"""
Token usage tracking and attribution.

The core idea: token/compute cost is meaningless in aggregate ("we spent
$40K on AI last month") and only becomes actionable once it's attributed
to a dimension someone can act on — a task type ("we spent $28K on
support_query tasks that could mostly run on a cheaper model") or a tag
("the agentic-commerce initiative is 60% of our frontier-model spend").
This module is the attribution layer for both.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .models import GovernedTask, PriceTable, TaskType


@dataclass
class TaskTypeUsageSummary:
    task_type: TaskType
    task_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def avg_cost_per_task(self) -> float:
        return self.total_cost_usd / self.task_count if self.task_count else 0.0


@dataclass
class TagUsageSummary:
    """Cost rolled up against one tag key/value pair.

    Pure aggregation — the tracker groups by whatever tags callers
    attached, without interpreting what those tags mean.
    """

    tag_key: str
    tag_value: str
    task_count: int = 0
    total_cost_usd: float = 0.0
    escalated_count: int = 0
    models_used: dict[str, int] = field(default_factory=dict)

    @property
    def avg_cost_per_task(self) -> float:
        return self.total_cost_usd / self.task_count if self.task_count else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.escalated_count / self.task_count if self.task_count else 0.0


class TokenTracker:
    """Aggregates token usage and cost across a batch of governed tasks."""

    def __init__(self, price_table: PriceTable | None = None):
        self.price_table = price_table or PriceTable()
        self._summaries: dict[TaskType, TaskTypeUsageSummary] = defaultdict(
            lambda: TaskTypeUsageSummary(task_type=TaskType.UNCLASSIFIED)
        )
        self._tag_summaries: dict[tuple[str, str], TagUsageSummary] = {}

    def record(self, task: GovernedTask) -> float:
        """Record a completed task's usage and return its cost in USD."""
        if task.token_usage is None:
            return 0.0

        cost = task.token_usage.cost_usd(self.price_table)

        summary = self._summaries[task.task_type]
        summary.task_type = task.task_type
        summary.task_count += 1
        summary.total_input_tokens += task.token_usage.input_tokens
        summary.total_output_tokens += task.token_usage.output_tokens
        summary.total_cost_usd += cost

        model = task.token_usage.model_name
        escalated = bool(task.metadata.get("escalated"))
        for key, value in task.tags.items():
            tag_summary = self._tag_summaries.setdefault(
                (key, value), TagUsageSummary(tag_key=key, tag_value=value)
            )
            tag_summary.task_count += 1
            tag_summary.total_cost_usd += cost
            if escalated:
                tag_summary.escalated_count += 1
            tag_summary.models_used[model] = tag_summary.models_used.get(model, 0) + 1

        return cost

    def summaries(self) -> list[TaskTypeUsageSummary]:
        return sorted(
            self._summaries.values(), key=lambda s: s.total_cost_usd, reverse=True
        )

    def tag_summaries(self, tag_key: str | None = None) -> list[TagUsageSummary]:
        """Cost rolled up by tag, optionally filtered to a single tag key."""
        values = [
            s for s in self._tag_summaries.values()
            if tag_key is None or s.tag_key == tag_key
        ]
        return sorted(values, key=lambda s: s.total_cost_usd, reverse=True)

    def total_cost_usd(self) -> float:
        return sum(s.total_cost_usd for s in self._summaries.values())
