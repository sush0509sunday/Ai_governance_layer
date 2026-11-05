"""
Route AI work on the work item first and the prompt second.

Most AI requests inside a software organization are attached to a work
item, such as a Jira issue, that already says what kind of work it is.
Its fields are a stronger routing signal than free text, and they are
much harder to game: triage sets the issue type and component, not the
person typing the prompt.

`WorkItemRouter` uses those fields to set the task type and baseline
model tier, then lets the prompt classifier refine within that choice:

* The work item's issue type (or a label) decides the task type and the
  baseline tier.
* Its component decides risk. High-risk components (for example auth or
  billing) are bumped to the strongest tier and flagged for human review.
* The prompt classifier only refines. The work item is trusted by default;
  the request is escalated one tier only when the prompt confidently points
  to a different kind of work (say, a support ticket that asks for a code
  review). The disagreement is recorded, never silently resolved.
* Every work-item field becomes a cost-attribution tag, so spend rolls up
  by project, component, issue type, epic and requester for finance.

Work items with no matching rule fall back to prompt-only routing through
the wrapped `AgentOrchestrator`.

How fields arrive (a Jira webhook, the REST API, a CI pipeline) is left to
the caller: build a `WorkItem` and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent_orchestrator import AgentOrchestrator, _ROUTING_POLICY
from .models import ExecutionMode, GovernedTask, TaskType


@dataclass
class WorkItem:
    """The routing-relevant fields of a work item, such as a Jira issue."""

    key: str                       # e.g. "CHECKOUT-142"
    issue_type: str                # e.g. "Bug", "Story", "Task", "Service Request"
    project: str | None = None
    component: str | None = None
    epic: str | None = None
    labels: list[str] = field(default_factory=list)
    requester: str | None = None

    def tags(self) -> dict[str, str]:
        tags = {"work_item": self.key, "issue_type": self.issue_type.lower()}
        for name in ("project", "component", "epic", "requester"):
            value = getattr(self, name)
            if value:
                tags[name] = value
        return tags


@dataclass
class WorkItemPolicy:
    """Maps work-item fields to a task type, a baseline tier and a risk level.

    Lookups are case-insensitive. Labels are checked before the issue type,
    because a label such as "docs" is more specific than "Task".
    """

    by_label: dict[str, TaskType] = field(default_factory=dict)
    by_issue_type: dict[str, TaskType] = field(default_factory=dict)
    high_risk_components: set[str] = field(default_factory=set)
    high_risk_model: str = "frontier-model"
    # Baseline tier per task type on the work-item path, where it differs
    # from the prompt-only routing policy.
    baseline_overrides: dict[TaskType, str] = field(default_factory=dict)
    # Prompt labels that are consistent with a work item's task type, e.g. a
    # bug report reads like a support question. These never count as conflict.
    compatible: dict[TaskType, set[TaskType]] = field(default_factory=dict)

    def baseline_for(self, task_type: TaskType) -> str:
        return self.baseline_overrides.get(task_type, _ROUTING_POLICY.get(task_type, "frontier-model"))

    def conflicts(self, work_item_type: TaskType, prompt_type: TaskType) -> bool:
        if prompt_type in (work_item_type, TaskType.UNCLASSIFIED):
            return False
        return prompt_type not in self.compatible.get(work_item_type, set())

    def task_type_for(self, item: WorkItem) -> TaskType | None:
        labels = {label.lower() for label in item.labels}
        for label, task_type in self.by_label.items():
            if label.lower() in labels:
                return task_type
        for issue_type, task_type in self.by_issue_type.items():
            if issue_type.lower() == item.issue_type.lower():
                return task_type
        return None

    def is_high_risk(self, item: WorkItem) -> bool:
        if not item.component:
            return False
        return item.component.lower() in {c.lower() for c in self.high_risk_components}


def default_sdlc_policy() -> WorkItemPolicy:
    """A starting policy for common SDLC work. Calibrate it on your own traffic."""
    return WorkItemPolicy(
        by_label={
            "code-review": TaskType.CODE_REVIEW,
            "docs": TaskType.DOCUMENTATION,
            "tests": TaskType.TEST_GENERATION,
        },
        by_issue_type={
            "Service Request": TaskType.SUPPORT_QUERY,
            "Support": TaskType.SUPPORT_QUERY,
            "Bug": TaskType.BUG_TRIAGE,
            "Incident": TaskType.BUG_TRIAGE,
        },
        high_risk_components={"auth", "billing", "security"},
        baseline_overrides={TaskType.CODE_REVIEW: "mid-tier-model"},
        compatible={
            TaskType.BUG_TRIAGE: {TaskType.SUPPORT_QUERY, TaskType.DATA_LOOKUP},
            TaskType.SUPPORT_QUERY: {TaskType.DATA_LOOKUP},
            TaskType.DOCUMENTATION: {TaskType.CONTENT_DRAFTING, TaskType.REPORT_GENERATION},
            TaskType.TEST_GENERATION: {TaskType.CODE_REVIEW},
        },
    )


class WorkItemRouter:
    """Routes a request using its work item first and its prompt second."""

    def __init__(self, orchestrator: AgentOrchestrator, policy: WorkItemPolicy | None = None):
        self.orchestrator = orchestrator
        self.policy = policy or default_sdlc_policy()

    def route(self, item: WorkItem, prompt: str, extra_tags: dict[str, str] | None = None) -> GovernedTask:
        tags = {**item.tags(), **(extra_tags or {})}
        task_type = self.policy.task_type_for(item)

        if task_type is None:
            # No rule for this work item: fall back to prompt-only routing.
            task = self.orchestrator.process(item.key, prompt, tags=tags)
            task.metadata["routing_signal"] = "prompt_only"
            return task

        classification = self.orchestrator.classifier.classify(prompt)
        task = GovernedTask(
            task_id=item.key,
            raw_text=prompt,
            task_type=task_type,
            classification_confidence=classification.confidence,
            tags=tags,
        )
        task.metadata["routing_signal"] = "work_item"
        task.metadata["classification_method"] = classification.method
        task.metadata["prompt_task_type"] = classification.task_type.value

        base_model = self.policy.baseline_for(task_type)
        high_risk = self.policy.is_high_risk(item)
        if high_risk:
            base_model = self.policy.high_risk_model
            task.metadata["human_review"] = True

        # The work item is trusted by default. Escalate only when the prompt
        # confidently points to a different kind of work; the structured field
        # still decides the task type, the prompt only buys caution.
        confident = classification.confidence >= self.orchestrator.config.escalation_confidence_threshold
        conflict = confident and self.policy.conflicts(task_type, classification.task_type)
        model = base_model
        if conflict:
            model = self.orchestrator._escalate(base_model)
        escalated = model != base_model
        if conflict:
            task.metadata["signal_conflict"] = True

        task.execution_mode = ExecutionMode.GOVERNED_AGENT
        task.token_usage = self.orchestrator._simulate_usage(prompt, model)
        task.metadata["routed_model"] = model
        task.metadata["baseline_model"] = base_model
        task.metadata["escalated"] = escalated
        return task

