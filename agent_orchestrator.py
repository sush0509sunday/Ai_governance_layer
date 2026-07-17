"""
Background agent orchestration.

Once a task is classified, the orchestrator decides *how* it gets
executed: handed to a human, run through an ungoverned ad-hoc AI call, or
routed to a governed background agent that operates within the cost and
model-selection rules the governance layer defines. This module is a
lightweight, dependency-free simulation of that routing decision — in a
real system, GOVERNED_AGENT execution would call out to actual model
APIs and real task-runner infrastructure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .models import ExecutionMode, GovernedTask, PriceTable, TaskType, TokenUsage
from .task_classifier import ClassificationResult, TaskClassifier

# Which model tier a governed agent should use per task type. This is the
# "do more with less" routing decision: simple, well-understood task
# types go to cheap/fast models; only ambiguous or high-stakes task types
# go to the expensive frontier model.
_ROUTING_POLICY: dict[TaskType, str] = {
    TaskType.SUPPORT_QUERY: "small-fast-model",
    TaskType.DATA_LOOKUP: "small-fast-model",
    TaskType.CONTENT_DRAFTING: "mid-tier-model",
    TaskType.REPORT_GENERATION: "mid-tier-model",
    TaskType.CODE_REVIEW: "frontier-model",
    TaskType.WORKFLOW_AUTOMATION: "mid-tier-model",
    TaskType.UNCLASSIFIED: "frontier-model",  # be conservative when unsure
}


@dataclass
class OrchestratorConfig:
    classification_confidence_threshold: float = 0.6
    rng_seed: int | None = 42


class AgentOrchestrator:
    """Routes classified tasks to an execution mode and simulates their cost."""

    def __init__(self, classifier: TaskClassifier | None = None, config: OrchestratorConfig | None = None):
        self.classifier = classifier or TaskClassifier()
        self.config = config or OrchestratorConfig()
        self._rng = random.Random(self.config.rng_seed)

    def process(self, task_id: str, raw_text: str) -> GovernedTask:
        classification: ClassificationResult = self.classifier.classify(raw_text)

        task = GovernedTask(
            task_id=task_id,
            raw_text=raw_text,
            task_type=classification.task_type,
            classification_confidence=classification.confidence,
        )

        if classification.confidence < self.config.classification_confidence_threshold:
            # Low-confidence classification: don't hand this to an
            # unsupervised background agent. Fall back to a human-in-the-loop
            # (ungoverned manual) path rather than guessing.
            task.execution_mode = ExecutionMode.UNGOVERNED_MANUAL
            task.metadata["reason"] = "low_confidence_fallback_to_manual"
            return task

        model = _ROUTING_POLICY.get(classification.task_type, "frontier-model")
        task.execution_mode = ExecutionMode.GOVERNED_AGENT
        task.token_usage = self._simulate_usage(raw_text, model)
        task.metadata["routed_model"] = model
        return task

    def _simulate_usage(self, raw_text: str, model: str) -> TokenUsage:
        """Simulate realistic-looking token counts for a demo run.

        Replace this with real usage figures returned by an actual model
        call in a production integration.
        """
        base_input = max(30, len(raw_text.split()) * 4)
        input_tokens = base_input + self._rng.randint(0, 60)
        output_tokens = self._rng.randint(80, 400)
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, model_name=model)
