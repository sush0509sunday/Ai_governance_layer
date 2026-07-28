"""
Background agent orchestration.

Once a task is classified, the orchestrator decides *how* it gets
executed: handed to a human, escalated to a pricier model tier because
the classifier wasn't confident enough to trust the cheap path, or run
at its normal governed tier. This module is a lightweight,
dependency-free simulation of that routing decision — in a real system,
GOVERNED_AGENT execution would call out to actual model APIs and real
task-runner infrastructure.

Three governance behaviors live here, each addressing a specific failure
mode common in routing systems:

1. Confidence-tiered escalation. A single hard cutoff between "trust it"
   and "don't" throws away information. Below a manual-fallback floor, a
   task goes to a human — the classifier isn't trusted at all. Between
   that floor and a higher escalation threshold, the task type is trusted
   but the classification confidence isn't high enough to risk running it
   on the cheap model tier, so it's bumped up one tier instead.
2. Overlap / multi-intent detection. If the top two candidate task types
   score within a small margin of each other, the request is flagged
   rather than having the orchestrator silently pick one and hope.
3. Deterministic, swappable classification. The orchestrator depends on
   `BaseClassifier`, not any specific implementation, so rule-based,
   embedding-based, LLM-based, or ensemble classifiers are all drop-in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .models import ExecutionMode, GovernedTask, TaskType, TokenUsage
from .task_classifier import BaseClassifier, ClassificationResult, RuleBasedClassifier

# Which model tier a governed agent should use per task type, before any
# confidence-based escalation. This is the "do more with less" routing
# decision: simple, well-understood task types go to cheap/fast models;
# only ambiguous or high-stakes task types go to the expensive frontier
# model.
_ROUTING_POLICY: dict[TaskType, str] = {
    TaskType.SUPPORT_QUERY: "small-fast-model",
    TaskType.DATA_LOOKUP: "small-fast-model",
    TaskType.CONTENT_DRAFTING: "mid-tier-model",
    TaskType.REPORT_GENERATION: "mid-tier-model",
    TaskType.CODE_REVIEW: "frontier-model",
    TaskType.WORKFLOW_AUTOMATION: "mid-tier-model",
    TaskType.UNCLASSIFIED: "frontier-model",  # be conservative when unsure
}

# Ordered cheapest -> most expensive. Used for confidence-based
# escalation: if the classifier's confidence doesn't clear the
# escalation threshold, step the request up one tier rather than run it
# on the cheap model or immediately punt to a human.
_TIER_ORDER = ["small-fast-model", "mid-tier-model", "frontier-model"]


@dataclass
class OrchestratorConfig:
    # Below this, the classification isn't trusted at all — the task
    # goes to a human rather than any AI path. Kept under its original
    # name for backward compatibility with existing callers/tests.
    classification_confidence_threshold: float = 0.6
    # Between classification_confidence_threshold and this value, the
    # task type is trusted but confidence isn't high enough to risk the
    # baseline (often cheaper) model tier, so the request is escalated
    # one tier up. At or above this value, the baseline tier is used.
    #
    # This number must be calibrated against real traffic, not guessed.
    # Classifier confidence tends to cluster rather than spread evenly,
    # so a threshold that lands inside a dense cluster makes escalation
    # rate — and therefore spend — hypersensitive to a change in the
    # third decimal place. Use `escalation_sensitivity()` below to see
    # where the cliffs are for your own data before setting this.
    escalation_confidence_threshold: float = 0.75
    # If the top two candidate task types score within this margin of
    # each other, treat the request as overlapping/multi-intent.
    overlap_margin: float = 0.1
    rng_seed: int | None = 42


class AgentOrchestrator:
    """Routes classified tasks to an execution mode and simulates their cost."""

    def __init__(self, classifier: BaseClassifier | None = None, config: OrchestratorConfig | None = None):
        self.classifier = classifier or RuleBasedClassifier()
        self.config = config or OrchestratorConfig()
        self._rng = random.Random(self.config.rng_seed)

    def process(
        self,
        task_id: str,
        raw_text: str,
        tags: dict[str, str] | None = None,
    ) -> GovernedTask:
        classification: ClassificationResult = self.classifier.classify(raw_text)

        task = GovernedTask(
            task_id=task_id,
            raw_text=raw_text,
            task_type=classification.task_type,
            classification_confidence=classification.confidence,
            tags=dict(tags or {}),
        )
        task.metadata["classification_method"] = classification.method

        if self._is_overlapping(classification):
            top_two = sorted(classification.scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
            task.metadata["multi_intent"] = True
            task.metadata["overlap_candidates"] = [t.value for t, _ in top_two]

        if classification.confidence < self.config.classification_confidence_threshold:
            # Low-confidence classification: don't hand this to an
            # unsupervised background agent. Fall back to a
            # human-in-the-loop (ungoverned manual) path rather than
            # guessing.
            task.execution_mode = ExecutionMode.UNGOVERNED_MANUAL
            task.metadata["reason"] = "low_confidence_fallback_to_manual"
            return task

        base_model = _ROUTING_POLICY.get(classification.task_type, "frontier-model")
        model = base_model
        escalated = False
        if classification.confidence < self.config.escalation_confidence_threshold:
            model = self._escalate(base_model)
            escalated = model != base_model

        task.execution_mode = ExecutionMode.GOVERNED_AGENT
        task.token_usage = self._simulate_usage(raw_text, model)
        task.metadata["routed_model"] = model
        task.metadata["escalated"] = escalated
        if escalated:
            task.metadata["baseline_model"] = base_model
        return task

    def _escalate(self, model: str) -> str:
        if model not in _TIER_ORDER:
            return model
        idx = _TIER_ORDER.index(model)
        return _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]

    def _is_overlapping(self, classification: ClassificationResult) -> bool:
        if len(classification.scores) < 2:
            return False
        ranked = sorted(classification.scores.values(), reverse=True)
        return (ranked[0] - ranked[1]) <= self.config.overlap_margin

    def _simulate_usage(self, raw_text: str, model: str) -> TokenUsage:
        """Simulate realistic-looking token counts for a demo run.

        Replace this with real usage figures returned by an actual model
        call in a production integration.
        """
        base_input = max(30, len(raw_text.split()) * 4)
        input_tokens = base_input + self._rng.randint(0, 60)
        output_tokens = self._rng.randint(80, 400)
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, model_name=model)


def escalation_sensitivity(
    classifier: BaseClassifier,
    texts: list[str],
    thresholds: list[float] | None = None,
    manual_floor: float = 0.6,
) -> list[dict]:
    """Show how escalation rate responds to the escalation threshold.

    Escalation is the single knob with the most direct effect on spend:
    every escalated request runs on a pricier model tier than its task
    type would otherwise get. Setting that threshold by intuition is how
    a governance layer ends up escalating most of its traffic and costing
    more than no governance at all.

    Classifier confidence rarely spreads evenly — it clusters. If a
    threshold lands inside a cluster, escalation rate can swing from near
    zero to near half on a change too small to notice by eye. This
    function makes those cliffs visible: run it over a representative
    sample of real requests, look for a threshold sitting in a *gap*
    between clusters rather than on top of one, and set the config from
    that evidence.

    Returns one row per candidate threshold with the share of requests
    that would be escalated, sent to a human, or run at baseline tier.
    """
    if thresholds is None:
        thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    confidences = [classifier.classify(t).confidence for t in texts]
    total = len(confidences) or 1

    rows = []
    for threshold in thresholds:
        manual = sum(1 for c in confidences if c < manual_floor)
        escalated = sum(1 for c in confidences if manual_floor <= c < threshold)
        baseline = total - manual - escalated
        rows.append({
            "threshold": threshold,
            "manual_rate": round(manual / total, 4),
            "escalation_rate": round(escalated / total, 4),
            "baseline_rate": round(baseline / total, 4),
        })
    return rows
