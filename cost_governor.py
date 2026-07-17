"""
Cost governance: budgets, alerts, and avoided-cost modeling.

This is the piece that turns raw usage data into decisions: which task
types are over budget, which ones are candidates for routing to a
cheaper model, and — importantly for measuring impact — what the
*avoided* cost is compared to an ungoverned baseline.

The avoided-cost model here is intentionally simple and clearly labeled
as an estimate. In a real petition or business case, this is exactly the
kind of number that must be presented as a projection/methodology, not
as an audited, realized result, unless it has actually been measured
against real historical spend.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import TaskType
from .token_tracker import TaskTypeUsageSummary, TokenTracker


@dataclass
class Budget:
    task_type: TaskType
    monthly_cap_usd: float


@dataclass
class BudgetAlert:
    task_type: TaskType
    spend_usd: float
    cap_usd: float
    pct_of_cap: float

    @property
    def is_over_cap(self) -> bool:
        return self.pct_of_cap >= 1.0


class CostGovernor:
    """Applies budgets to tracked usage and models avoided cost."""

    def __init__(self, budgets: list[Budget] | None = None):
        self.budgets: dict[TaskType, Budget] = {b.task_type: b for b in (budgets or [])}

    def check_budgets(self, tracker: TokenTracker) -> list[BudgetAlert]:
        alerts = []
        for summary in tracker.summaries():
            budget = self.budgets.get(summary.task_type)
            if budget is None:
                continue
            pct = summary.total_cost_usd / budget.monthly_cap_usd if budget.monthly_cap_usd else 0.0
            alerts.append(
                BudgetAlert(
                    task_type=summary.task_type,
                    spend_usd=round(summary.total_cost_usd, 2),
                    cap_usd=budget.monthly_cap_usd,
                    pct_of_cap=round(pct, 2),
                )
            )
        return sorted(alerts, key=lambda a: a.pct_of_cap, reverse=True)

    @staticmethod
    def estimate_avoided_cost(
        summary: TaskTypeUsageSummary,
        ungoverned_multiplier: float = 2.4,
    ) -> float:
        """Estimate what this task type would have cost without governance.

        `ungoverned_multiplier` represents the combined effect of things a
        governance layer typically prevents: redundant context on every
        call, no semantic caching, no routing to cheaper models for
        simple tasks, and unmanaged agent loops. 2.4x is a illustrative,
        conservative placeholder — a real implementation would derive
        this from an actual before/after measurement, not assume it.
        """
        governed_cost = summary.total_cost_usd
        hypothetical_ungoverned_cost = governed_cost * ungoverned_multiplier
        return round(hypothetical_ungoverned_cost - governed_cost, 2)
