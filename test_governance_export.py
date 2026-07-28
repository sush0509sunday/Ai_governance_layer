import json

import pytest

from src.cost_governor import Budget, CostGovernor, TagBudget
from src.governance_export import SCHEMA_VERSION, ExportBuilder, GovernanceFeed
from src.models import ExecutionMode, GovernedTask, TaskType, TokenUsage
from src.observability import DecisionLogger
from src.token_tracker import TokenTracker


def make_task(task_id="T1", task_type=TaskType.CODE_REVIEW, tags=None, escalated=False):
    return GovernedTask(
        task_id=task_id,
        raw_text="dummy",
        task_type=task_type,
        classification_confidence=0.85,
        execution_mode=ExecutionMode.GOVERNED_AGENT,
        token_usage=TokenUsage(1000, 2000, "mid-tier-model"),
        tags=tags or {},
        metadata={
            "escalated": escalated,
            "routed_model": "mid-tier-model",
            "classification_method": "ensemble",
        },
    )


@pytest.fixture
def wired():
    tracker = TokenTracker()
    logger = DecisionLogger()
    a = {"initiative": "agentic_commerce"}
    b = {"initiative": "internal_tools"}
    for i, tags in enumerate([a, a, a, b]):
        task = make_task(f"T{i}", tags=tags, escalated=(i == 0))
        tracker.record(task)
        logger.record(task)

    governor = CostGovernor(
        budgets=[Budget(TaskType.CODE_REVIEW, monthly_cap_usd=1.0)],
        tag_budgets=[
            TagBudget("initiative", "agentic_commerce", monthly_cap_usd=1.0),
            TagBudget("initiative", "internal_tools", monthly_cap_usd=0.00001),
        ],
    )
    return tracker, governor, logger


def test_document_is_versioned(wired):
    doc = ExportBuilder(*wired).build()
    assert doc["schema_version"] == SCHEMA_VERSION
    assert "generated_at" in doc


def test_document_has_all_three_grains(wired):
    doc = ExportBuilder(*wired).build()
    assert doc["summaries"]["by_task_type"]
    assert doc["summaries"]["by_tag"]
    assert doc["alerts"]
    assert doc["decisions"]


def test_projections_are_labeled_in_the_data(wired):
    """A projected number must be self-describing, not just documented."""
    doc = ExportBuilder(*wired).build()
    assert doc["totals"]["is_projection"] is True
    assert "PROJECTION" in doc["totals"]["projection_methodology"]
    for row in doc["summaries"]["by_task_type"]:
        assert row["is_projection"] is True


def test_aggregations_are_precomputed(wired):
    """Consumers must never need to re-derive cost math."""
    tracker = wired[0]
    doc = ExportBuilder(*wired).build()
    assert doc["totals"]["total_governed_cost_usd"] == round(tracker.total_cost_usd(), 4)
    row = doc["summaries"]["by_task_type"][0]
    assert row["avg_cost_per_task_usd"] > 0


def test_alerts_are_sorted_worst_first(wired):
    doc = ExportBuilder(*wired).build()
    pcts = [a["pct_of_cap"] for a in doc["alerts"]]
    assert pcts == sorted(pcts, reverse=True)


def test_decisions_can_be_omitted(wired):
    doc = ExportBuilder(*wired).build(include_decisions=False)
    assert "decisions" not in doc
    assert doc["summaries"]["by_task_type"]


def test_routing_health_included_when_logger_present(wired):
    doc = ExportBuilder(*wired).build()
    assert doc["routing_health"]["decisions"] == 4


def test_export_works_without_logger(wired):
    tracker, governor, _ = wired
    doc = ExportBuilder(tracker, governor).build()
    assert "routing_health" not in doc
    assert "decisions" not in doc


def test_write_and_read_round_trip(wired, tmp_path):
    path = ExportBuilder(*wired).write(tmp_path / "governance_export.json")
    feed = GovernanceFeed.from_file(path)

    assert feed.schema_version == SCHEMA_VERSION
    assert len(feed.decisions()) == 4
    assert len(feed.by_tag("initiative")) == 2
    assert feed.totals()["is_projection"] is True
    # File must be valid standalone JSON for non-Python consumers.
    json.loads(path.read_text())


def test_feed_filters_decisions_by_tag(wired):
    feed = GovernanceFeed(ExportBuilder(*wired).build())
    assert len(feed.decisions(initiative="agentic_commerce")) == 3
    assert len(feed.decisions(initiative="internal_tools")) == 1


def test_feed_can_return_only_breaches(wired):
    feed = GovernanceFeed(ExportBuilder(*wired).build())
    breaches = feed.alerts(over_cap_only=True)
    assert breaches
    assert all(a["status"] == "over_cap" for a in breaches)


def test_feed_rejects_missing_version():
    with pytest.raises(ValueError, match="schema_version"):
        GovernanceFeed({"summaries": {}})


def test_feed_rejects_incompatible_major_version():
    with pytest.raises(ValueError, match="Incompatible"):
        GovernanceFeed({"schema_version": "99.0", "summaries": {}})


def test_feed_accepts_compatible_minor_version(wired):
    doc = ExportBuilder(*wired).build()
    doc["schema_version"] = f"{SCHEMA_VERSION.split('.')[0]}.99"
    GovernanceFeed(doc)  # must not raise
