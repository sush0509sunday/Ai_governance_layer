import json

from src.agent_orchestrator import AgentOrchestrator, OrchestratorConfig
from src.models import ExecutionMode, GovernedTask, TaskType, TokenUsage
from src.observability import DecisionLogger


def governed(task_id="T1", escalated=False, multi_intent=False, tags=None):
    return GovernedTask(
        task_id=task_id,
        raw_text="dummy",
        task_type=TaskType.CODE_REVIEW,
        classification_confidence=0.8,
        execution_mode=ExecutionMode.GOVERNED_AGENT,
        token_usage=TokenUsage(100, 200, "mid-tier-model"),
        tags=tags or {},
        metadata={
            "escalated": escalated,
            "multi_intent": multi_intent,
            "routed_model": "mid-tier-model",
            "classification_method": "rule_based",
        },
    )


def manual(task_id="T2", tags=None):
    return GovernedTask(
        task_id=task_id,
        raw_text="dummy",
        task_type=TaskType.UNCLASSIFIED,
        execution_mode=ExecutionMode.UNGOVERNED_MANUAL,
        tags=tags or {},
        metadata={"reason": "low_confidence_fallback_to_manual"},
    )


def test_logger_records_one_row_per_task():
    logger = DecisionLogger()
    logger.record(governed())
    logger.record(manual())
    assert len(logger) == 2


def test_record_captures_cost_and_model():
    logger = DecisionLogger()
    record = logger.record(governed())
    assert record.routed_model == "mid-tier-model"
    assert record.cost_usd > 0
    assert record.classification_method == "rule_based"


def test_manual_fallback_record_has_zero_cost():
    logger = DecisionLogger()
    record = logger.record(manual())
    assert record.cost_usd == 0.0
    assert record.routed_model is None
    assert record.reason == "low_confidence_fallback_to_manual"


def test_escalation_rate():
    logger = DecisionLogger()
    logger.record(governed("T1", escalated=True))
    logger.record(governed("T2", escalated=False))
    logger.record(governed("T3", escalated=False))
    logger.record(governed("T4", escalated=False))
    assert logger.escalation_rate() == 0.25


def test_manual_fallback_rate():
    logger = DecisionLogger()
    logger.record(governed("T1"))
    logger.record(manual("T2"))
    assert logger.manual_fallback_rate() == 0.5


def test_multi_intent_rate():
    logger = DecisionLogger()
    logger.record(governed("T1", multi_intent=True))
    logger.record(governed("T2", multi_intent=False))
    assert logger.multi_intent_rate() == 0.5


def test_rates_are_zero_on_empty_logger():
    logger = DecisionLogger()
    assert logger.escalation_rate() == 0.0
    assert logger.manual_fallback_rate() == 0.0
    assert logger.multi_intent_rate() == 0.0


def test_metrics_can_be_filtered_by_tag():
    logger = DecisionLogger()
    a = {"initiative": "agentic_commerce"}
    b = {"initiative": "internal_tools"}
    logger.record(governed("T1", escalated=True, tags=a))
    logger.record(governed("T2", escalated=True, tags=a))
    logger.record(governed("T3", escalated=False, tags=b))

    assert logger.escalation_rate(initiative="agentic_commerce") == 1.0
    assert logger.escalation_rate(initiative="internal_tools") == 0.0
    assert len(logger.records(initiative="agentic_commerce")) == 2


def test_model_mix_counts_tiers():
    logger = DecisionLogger()
    logger.record(governed("T1"))
    logger.record(governed("T2"))
    logger.record(manual("T3"))
    assert logger.model_mix() == {"mid-tier-model": 2}


def test_health_snapshot_has_expected_keys():
    logger = DecisionLogger()
    logger.record(governed())
    snapshot = logger.health_snapshot()
    for key in (
        "decisions",
        "escalation_rate",
        "manual_fallback_rate",
        "multi_intent_rate",
        "model_mix",
    ):
        assert key in snapshot


def test_jsonl_round_trip(tmp_path):
    logger = DecisionLogger()
    logger.record(governed("T1", tags={"initiative": "agentic_commerce"}))
    logger.record(manual("T2"))

    path = logger.write_jsonl(tmp_path / "decisions.jsonl")
    loaded = DecisionLogger.load_jsonl(path)

    assert len(loaded) == 2
    assert loaded[0].task_id == "T1"
    assert loaded[0].tags == {"initiative": "agentic_commerce"}
    # Every line must be independently parseable JSON.
    for line in path.read_text().splitlines():
        json.loads(line)


def test_logger_integrates_with_orchestrator():
    """The logger must capture what the orchestrator actually produced."""
    orchestrator = AgentOrchestrator(
        config=OrchestratorConfig(
            classification_confidence_threshold=0.1,
            escalation_confidence_threshold=0.99,
        )
    )
    logger = DecisionLogger()
    task = orchestrator.process("T1", "help", tags={"initiative": "internal_tools"})
    record = logger.record(task)

    assert record.escalated is True
    assert record.baseline_model == "small-fast-model"
    assert record.tags == {"initiative": "internal_tools"}
