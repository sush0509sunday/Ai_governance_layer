from src.agent_orchestrator import (
    AgentOrchestrator,
    OrchestratorConfig,
    escalation_sensitivity,
)
from src.models import ExecutionMode, TaskType
from src.task_classifier import RuleBasedClassifier


def test_high_confidence_task_routes_to_governed_agent():
    orchestrator = AgentOrchestrator()
    task = orchestrator.process("T1", "please review this pull request before merge")
    assert task.execution_mode == ExecutionMode.GOVERNED_AGENT
    assert task.token_usage is not None


def test_low_confidence_task_falls_back_to_manual():
    orchestrator = AgentOrchestrator(
        config=OrchestratorConfig(classification_confidence_threshold=0.99)
    )
    task = orchestrator.process("T1", "please review this pull request before merge")
    assert task.execution_mode == ExecutionMode.UNGOVERNED_MANUAL
    assert task.token_usage is None
    assert task.metadata["reason"] == "low_confidence_fallback_to_manual"


def test_unclassified_text_does_not_crash():
    orchestrator = AgentOrchestrator()
    task = orchestrator.process("T1", "lorem ipsum dolor sit amet")
    assert task.task_type == TaskType.UNCLASSIFIED
    assert task.execution_mode == ExecutionMode.UNGOVERNED_MANUAL


def test_deterministic_with_fixed_seed():
    o1 = AgentOrchestrator(config=OrchestratorConfig(rng_seed=7))
    o2 = AgentOrchestrator(config=OrchestratorConfig(rng_seed=7))
    t1 = o1.process("T1", "automate this recurring background job")
    t2 = o2.process("T1", "automate this recurring background job")
    assert t1.token_usage.input_tokens == t2.token_usage.input_tokens
    assert t1.token_usage.output_tokens == t2.token_usage.output_tokens


# ---- escalation ------------------------------------------------------

def test_mid_confidence_escalates_one_tier():
    """Trusted category, shaky confidence -> step up a tier, don't punt."""
    orchestrator = AgentOrchestrator(
        config=OrchestratorConfig(
            classification_confidence_threshold=0.1,
            escalation_confidence_threshold=0.99,
        )
    )
    task = orchestrator.process("T1", "help")  # support_query, low-ish confidence
    assert task.execution_mode == ExecutionMode.GOVERNED_AGENT
    assert task.metadata["escalated"] is True
    assert task.metadata["baseline_model"] == "small-fast-model"
    assert task.metadata["routed_model"] == "mid-tier-model"


def test_high_confidence_uses_baseline_tier_without_escalating():
    orchestrator = AgentOrchestrator(
        config=OrchestratorConfig(escalation_confidence_threshold=0.0)
    )
    task = orchestrator.process("T1", "help, it's broken and not working")
    assert task.metadata["escalated"] is False
    assert task.metadata["routed_model"] == "small-fast-model"
    assert "baseline_model" not in task.metadata


def test_escalation_cannot_exceed_top_tier():
    orchestrator = AgentOrchestrator(
        config=OrchestratorConfig(
            classification_confidence_threshold=0.1,
            escalation_confidence_threshold=0.99,
        )
    )
    task = orchestrator.process("T1", "review this diff")  # already frontier tier
    assert task.metadata["routed_model"] == "frontier-model"


# ---- overlap / multi-intent -----------------------------------------

def test_overlapping_request_is_flagged():
    """Near-tied top-two categories must be surfaced, not silently resolved."""
    orchestrator = AgentOrchestrator()
    task = orchestrator.process("T1", "please review this diff and help me with the error")
    assert task.metadata.get("multi_intent") is True
    assert len(task.metadata["overlap_candidates"]) == 2


def test_clear_single_intent_is_not_flagged():
    orchestrator = AgentOrchestrator()
    task = orchestrator.process("T1", "automate schedule a recurring background job")
    assert task.metadata.get("multi_intent") is not True


# ---- tags ------------------------------------------------------------

def test_tags_are_attached_and_isolated():
    orchestrator = AgentOrchestrator()
    tags = {"initiative": "agentic_commerce"}
    task = orchestrator.process("T1", "review this diff", tags=tags)
    assert task.tags == {"initiative": "agentic_commerce"}
    # Mutating the caller's dict must not mutate the task's copy.
    tags["initiative"] = "changed"
    assert task.tags["initiative"] == "agentic_commerce"


def test_tags_default_to_empty():
    task = AgentOrchestrator().process("T1", "review this diff")
    assert task.tags == {}


def test_classification_method_is_recorded():
    task = AgentOrchestrator().process("T1", "review this diff")
    assert task.metadata["classification_method"] == "rule_based"


# ---- threshold calibration -------------------------------------------

TEXTS = [
    "please review this pull request diff",
    "help, it's broken and not working",
    "how many active users last week",
    "automate this recurring background job",
    "the sky is a nice color today",
]


def test_escalation_sensitivity_rates_sum_to_one():
    for row in escalation_sensitivity(RuleBasedClassifier(), TEXTS):
        total = row["manual_rate"] + row["escalation_rate"] + row["baseline_rate"]
        assert abs(total - 1.0) < 1e-6


def test_escalation_rate_is_monotonic_in_threshold():
    """Raising the threshold can only escalate more, never less."""
    rows = escalation_sensitivity(RuleBasedClassifier(), TEXTS)
    rates = [r["escalation_rate"] for r in rows]
    assert rates == sorted(rates)


def test_escalation_sensitivity_accepts_custom_thresholds():
    rows = escalation_sensitivity(RuleBasedClassifier(), TEXTS, thresholds=[0.6, 0.9])
    assert [r["threshold"] for r in rows] == [0.6, 0.9]


def test_escalation_sensitivity_handles_empty_input():
    rows = escalation_sensitivity(RuleBasedClassifier(), [])
    assert all(r["escalation_rate"] == 0.0 for r in rows)
