"""
Token usage tracking and attribution.

The core idea: token/compute cost is meaningless in aggregate ("we spent
$40K on AI last month") and only becomes actionable once it's attributed
to a task type ("we spent $28K on support_query tasks that could mostly
run on a cheaper model"). This module is the attribution layer.
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


class TokenTracker:
    """Aggregates token usage and cost across a batch of governed tasks."""

    def __init__(self, price_table: PriceTable | None = None):
        self.price_table = price_table or PriceTable()
        self._summaries: dict[TaskType, TaskTypeUsageSummary] = defaultdict(
            lambda: TaskTypeUsageSummary(task_type=TaskType.UNCLASSIFIED)
        )

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

        return cost

    def summaries(self) -> list[TaskTypeUsageSummary]:
        return sorted(
            self._summaries.values(), key=lambda s: s.total_cost_usd, reverse=True
        )

    def total_cost_usd(self) -> float:
        return sum(s.total_cost_usd for s in self._summaries.values())
