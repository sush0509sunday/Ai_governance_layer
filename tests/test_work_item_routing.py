from src.agent_orchestrator import AgentOrchestrator
from src.models import ExecutionMode, TaskType
from src.observability import DecisionLogger
from src.task_classifier import BaseClassifier, ClassificationResult
from src.token_tracker import TokenTracker
from src.work_item_routing import WorkItem, WorkItemRouter, default_sdlc_policy


class Fixed(BaseClassifier):
    method_name = "fixed"

    def __init__(self, task_type, confidence):
        self.task_type = task_type
        self.confidence = confidence

    def classify(self, raw_text):
        return ClassificationResult(
            task_type=self.task_type, confidence=self.confidence,
            matched_signals=[], method=self.method_name,
            scores={self.task_type: self.confidence},
        )


def router(task_type, confidence):
    return WorkItemRouter(AgentOrchestrator(classifier=Fixed(task_type, confidence)))


def test_issue_type_sets_task_type_and_baseline_tier():
    item = WorkItem(key="CHECKOUT-1", issue_type="Bug", project="CHECKOUT", component="search")
    task = router(TaskType.BUG_TRIAGE, 0.9).route(item, "checkout search returns 500")
    assert task.task_type == TaskType.BUG_TRIAGE
    assert task.metadata["routing_signal"] == "work_item"
    assert task.metadata["routed_model"] == "mid-tier-model"
    assert not task.metadata["escalated"]
    assert task.execution_mode == ExecutionMode.GOVERNED_AGENT


def test_label_is_more_specific_than_issue_type():
    item = WorkItem(key="PLAT-7", issue_type="Task", labels=["docs"])
    task = router(TaskType.DOCUMENTATION, 0.9).route(item, "document the retry API")
    assert task.task_type == TaskType.DOCUMENTATION
    assert task.metadata["routed_model"] == "small-fast-model"


def test_high_risk_component_goes_to_strongest_tier_with_human_review():
    item = WorkItem(key="CHECKOUT-2", issue_type="Task", component="auth", labels=["code-review"])
    task = router(TaskType.CODE_REVIEW, 0.95).route(item, "review this token refresh change")
    assert task.metadata["routed_model"] == "frontier-model"
    assert task.metadata["human_review"] is True


def test_prompt_disagreement_escalates_but_work_item_keeps_task_type():
    item = WorkItem(key="SUP-3", issue_type="Service Request")
    task = router(TaskType.CODE_REVIEW, 0.95).route(item, "can you review this diff for me")
    assert task.task_type == TaskType.SUPPORT_QUERY
    assert task.metadata["signal_conflict"] is True
    assert task.metadata["routed_model"] == "mid-tier-model"   # small -> one tier up
    assert task.metadata["escalated"] is True


def test_low_confidence_prompt_does_not_override_the_work_item():
    item = WorkItem(key="SUP-4", issue_type="Service Request")
    task = router(TaskType.CODE_REVIEW, 0.5).route(item, "hmm")
    assert task.task_type == TaskType.SUPPORT_QUERY
    assert task.metadata["routed_model"] == "small-fast-model"
    assert not task.metadata["escalated"]
    assert task.execution_mode == ExecutionMode.GOVERNED_AGENT


def test_compatible_prompt_label_is_not_a_conflict():
    item = WorkItem(key="CHECKOUT-5", issue_type="Bug", component="search")
    task = router(TaskType.SUPPORT_QUERY, 0.95).route(item, "search page throws an error")
    assert task.task_type == TaskType.BUG_TRIAGE
    assert "signal_conflict" not in task.metadata
    assert task.metadata["routed_model"] == "mid-tier-model"


def test_code_review_uses_mid_tier_unless_component_is_high_risk():
    normal = WorkItem(key="C-1", issue_type="Task", component="search", labels=["code-review"])
    risky = WorkItem(key="C-2", issue_type="Task", component="auth", labels=["code-review"])
    r = router(TaskType.CODE_REVIEW, 0.95)
    assert r.route(normal, "review this pull request").metadata["routed_model"] == "mid-tier-model"
    assert r.route(risky, "review this pull request").metadata["routed_model"] == "frontier-model"


def test_unknown_work_item_falls_back_to_prompt_only_routing():
    item = WorkItem(key="X-1", issue_type="Epic")
    task = router(TaskType.SUPPORT_QUERY, 0.9).route(item, "I can't log in")
    assert task.metadata["routing_signal"] == "prompt_only"
    assert task.task_type == TaskType.SUPPORT_QUERY


def test_work_item_fields_become_cost_attribution_tags():
    r = router(TaskType.BUG_TRIAGE, 0.9)
    tracker, logger = TokenTracker(), DecisionLogger()
    items = [
        WorkItem(key="CHECKOUT-10", issue_type="Bug", project="CHECKOUT", component="search", requester="ana"),
        WorkItem(key="CHECKOUT-11", issue_type="Bug", project="CHECKOUT", component="search", requester="ana"),
        WorkItem(key="PLAT-12", issue_type="Bug", project="PLATFORM", component="infra", requester="raj"),
    ]
    for item in items:
        task = r.route(item, "triage this failure")
        tracker.record(task)
        logger.record(task)

    by_project = {s.tag_value: s for s in tracker.tag_summaries("project")}
    assert by_project["CHECKOUT"].task_count == 2
    assert by_project["PLATFORM"].task_count == 1
    assert {s.tag_value for s in tracker.tag_summaries("work_item")} == {"CHECKOUT-10", "CHECKOUT-11", "PLAT-12"}
    assert logger.escalation_rate(requester="ana") == 0.0


def test_default_policy_is_case_insensitive():
    policy = default_sdlc_policy()
    assert policy.task_type_for(WorkItem(key="A-1", issue_type="bug")) == TaskType.BUG_TRIAGE
    assert policy.is_high_risk(WorkItem(key="A-2", issue_type="Task", component="Billing"))


def test_decision_log_keeps_lineage_and_governance_flags():
    logger = DecisionLogger()
    r = router(TaskType.CODE_REVIEW, 0.95)
    risky = WorkItem(key="CHECKOUT-9", issue_type="Task", project="CHECKOUT",
                     component="billing", labels=["code-review"])
    record = logger.record(r.route(risky, "review this refund change"))

    # Lineage: one row ties the work item to its task type, model and cost.
    assert record.task_id == "CHECKOUT-9"
    assert record.tags["work_item"] == "CHECKOUT-9"
    assert record.task_type == TaskType.CODE_REVIEW.value
    assert record.routed_model == "frontier-model"
    assert record.cost_usd > 0
    assert record.routing_signal == "work_item"
    # Governance: high-risk work is visibly sent to human review.
    assert record.human_review is True
    assert logger.human_review_rate(project="CHECKOUT") == 1.0
