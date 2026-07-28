"""
Core data models for the AI Task Governance Layer.

These represent the shape of a "unit of AI work" as it flows through the
system: a raw request comes in, gets classified into a task type, gets
executed (by a human workflow, an automation, or a background agent), and
its token/compute cost gets recorded and attributed back to that task type.

This is the foundation the rest of the governance layer builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    """Canonical task categories the classifier assigns requests to.

    Real deployments would derive this taxonomy from actual usage data
    rather than hardcoding it — this is a representative starting set.
    """

    SUPPORT_QUERY = "support_query"
    CODE_REVIEW = "code_review"
    DATA_LOOKUP = "data_lookup"
    REPORT_GENERATION = "report_generation"
    CONTENT_DRAFTING = "content_drafting"
    WORKFLOW_AUTOMATION = "workflow_automation"
    UNCLASSIFIED = "unclassified"


class ExecutionMode(str, Enum):
    """How a classified task actually got executed.

    The whole point of a governance layer is to shift volume from
    UNGOVERNED_MANUAL / UNGOVERNED_AI toward GOVERNED_AGENT over time,
    while keeping cost and quality visible at every step.
    """

    UNGOVERNED_MANUAL = "ungoverned_manual"        # a human did it, no AI involved
    UNGOVERNED_AI = "ungoverned_ai"                # AI was used with no cost/quality controls
    GOVERNED_AGENT = "governed_agent"              # AI used through the governance layer


@dataclass
class TokenUsage:
    """Token accounting for a single unit of AI work."""

    input_tokens: int
    output_tokens: int
    model_name: str = "unspecified"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, price_table: "PriceTable") -> float:
        return price_table.cost(self.model_name, self.input_tokens, self.output_tokens)


@dataclass
class PriceTable:
    """Per-model, per-million-token pricing.

    Input and output tokens are commonly priced very differently (output
    is typically several times more expensive), which is why cost
    governance can't just count tokens — it has to weight them correctly.

    Rates here are illustrative round numbers chosen to show the shape of
    the tiering, not any specific vendor's current price list. Replace
    `rates` with live pricing pulled from your providers.
    """

    # (input_price_per_million, output_price_per_million), in USD
    rates: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "small-fast-model": (0.25, 1.25),
        "mid-tier-model": (3.00, 15.00),
        "frontier-model": (8.00, 40.00),
        "unspecified": (3.00, 15.00),
    })

    def cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = self.rates.get(model_name, self.rates["unspecified"])
        return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


@dataclass
class GovernedTask:
    """A single request as it flows through the governance layer end to end.

    `tags` is deliberately an open string->string map that the engine
    never interprets. Callers use it to attach whatever dimensions their
    organization cares about — team, initiative, environment, cost
    center. The governance layer attributes cost against those tags and
    passes them through to the export contract, but it takes no opinion
    on what they mean. That keeps this engine reusable across
    organizations with different structures, and lets a downstream
    consumer (dashboard, FinOps tool, chargeback system) do the
    strategic interpretation.
    """

    task_id: str
    raw_text: str
    task_type: TaskType = TaskType.UNCLASSIFIED
    classification_confidence: float = 0.0
    execution_mode: ExecutionMode = ExecutionMode.UNGOVERNED_MANUAL
    token_usage: Optional[TokenUsage] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
