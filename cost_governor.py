"""
Cost governance: budgets, alerts, and avoided-cost modeling.

This is the piece that turns raw usage data into decisions: which task
types are over budget, which tagged initiatives are trending past their
cap, which work is a candidate for routing to a cheaper model, and —
importantly for measuring impact — what the *avoided* cost is compared
to an ungoverned baseline.

The avoided-cost model here is intentionally simple and clearly labeled
as an estimate. In a real business case, this is exactly the kind of
number that must be presented as a projection with its methodology
attached, not as an audited, realized result, unless it has actually
been measured against real historical spend.
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
class TagBudget:
    """A spend cap scoped to a tag value rather than a task type.

    This is how a budget gets attached to something an executive
    recognizes — a team, an initiative, a cost center — without the
    engine needing to know what that tag means.
    """

    tag_key: str
    tag_value: str
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


@dataclass
class TagBudgetAlert:
    tag_key: str
    tag_value: str
    spend_usd: float
    cap_usd: float
    pct_of_cap: float

    @property
    def is_over_cap(self) -> bool:
        return self.pct_of_cap >= 1.0


class CostGovernor:
    """Applies budgets to tracked usage and models avoided cost."""

    def __init__(
        self,
        budgets: list[Budget] | None = None,
        tag_budgets: list[TagBudget] | None = None,
    ):
        self.budgets: dict[TaskType, Budget] = {b.task_type: b for b in (budgets or [])}
        self.tag_budgets: dict[tuple[str, str], TagBudget] = {
            (b.tag_key, b.tag_value): b for b in (tag_budgets or [])
        }

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

    def check_tag_budgets(self, tracker: TokenTracker) -> list[TagBudgetAlert]:
        """Budget check at the tag grain — the exec-facing view.

        Returns one alert per tag budget that has a matching spend
        record, sorted worst-first so an exception-based consumer can
        take the top N without sorting again.
        """
        alerts = []
        for summary in tracker.tag_summaries():
            budget = self.tag_budgets.get((summary.tag_key, summary.tag_value))
            if budget is None:
                continue
            pct = summary.total_cost_usd / budget.monthly_cap_usd if budget.monthly_cap_usd else 0.0
            alerts.append(
                TagBudgetAlert(
                    tag_key=summary.tag_key,
                    tag_value=summary.tag_value,
                    spend_usd=round(summary.total_cost_usd, 4),
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
        simple tasks, and unmanaged agent loops. 2.4x is an illustrative,
        conservative placeholder — a real implementation would derive
        this from an actual before/after measurement, not assume it.

        Any figure produced by this method is a PROJECTION. It should be
        labeled as such wherever it is reported.
        """
        governed_cost = summary.total_cost_usd
        hypothetical_ungoverned_cost = governed_cost * ungoverned_multiplier
        return round(hypothetical_ungoverned_cost - governed_cost, 2)
