from src.cost_governor import Budget, CostGovernor, TagBudget
from src.models import ExecutionMode, GovernedTask, PriceTable, TaskType, TokenUsage
from src.token_tracker import TokenTracker


def make_task(
    task_type: TaskType,
    input_tokens: int,
    output_tokens: int,
    model="mid-tier-model",
    tags=None,
    escalated=False,
) -> GovernedTask:
    return GovernedTask(
        task_id="T1",
        raw_text="dummy",
        task_type=task_type,
        execution_mode=ExecutionMode.GOVERNED_AGENT,
        token_usage=TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, model_name=model
        ),
        tags=tags or {},
        metadata={"escalated": escalated},
    )


def test_token_tracker_aggregates_by_task_type():
    tracker = TokenTracker()
    tracker.record(make_task(TaskType.SUPPORT_QUERY, 100, 200))
    tracker.record(make_task(TaskType.SUPPORT_QUERY, 50, 100))
    tracker.record(make_task(TaskType.CODE_REVIEW, 500, 500))

    summaries = {s.task_type: s for s in tracker.summaries()}
    assert summaries[TaskType.SUPPORT_QUERY].task_count == 2
    assert summaries[TaskType.SUPPORT_QUERY].total_input_tokens == 150
    assert summaries[TaskType.CODE_REVIEW].task_count == 1


def test_price_table_weights_output_tokens_higher():
    pt = PriceTable()
    cost_output_heavy = pt.cost("mid-tier-model", input_tokens=0, output_tokens=1_000_000)
    cost_input_heavy = pt.cost("mid-tier-model", input_tokens=1_000_000, output_tokens=0)
    assert cost_output_heavy > cost_input_heavy


def test_unknown_model_falls_back_to_default_rate():
    pt = PriceTable()
    assert pt.cost("no-such-model", 1_000_000, 0) == pt.cost("unspecified", 1_000_000, 0)


def test_task_without_usage_costs_nothing():
    tracker = TokenTracker()
    task = GovernedTask(task_id="T1", raw_text="x", task_type=TaskType.SUPPORT_QUERY)
    assert tracker.record(task) == 0.0
    assert tracker.total_cost_usd() == 0.0


def test_budget_alert_flags_over_cap():
    tracker = TokenTracker()
    tracker.record(make_task(TaskType.CODE_REVIEW, 1_000_000, 1_000_000))

    governor = CostGovernor(budgets=[Budget(TaskType.CODE_REVIEW, monthly_cap_usd=1.0)])
    alerts = governor.check_budgets(tracker)

    assert len(alerts) == 1
    assert alerts[0].is_over_cap is True


def test_estimate_avoided_cost_is_nonnegative():
    tracker = TokenTracker()
    tracker.record(make_task(TaskType.SUPPORT_QUERY, 100, 200))
    summary = tracker.summaries()[0]
    assert CostGovernor.estimate_avoided_cost(summary) >= 0


# ---- tag-grain attribution -------------------------------------------

def test_tracker_rolls_up_cost_by_tag():
    tracker = TokenTracker()
    tags_a = {"initiative": "agentic_commerce"}
    tags_b = {"initiative": "internal_tools"}
    tracker.record(make_task(TaskType.CODE_REVIEW, 1000, 1000, tags=tags_a))
    tracker.record(make_task(TaskType.CODE_REVIEW, 1000, 1000, tags=tags_a))
    tracker.record(make_task(TaskType.DATA_LOOKUP, 100, 100, tags=tags_b))

    rollup = {t.tag_value: t for t in tracker.tag_summaries("initiative")}
    assert rollup["agentic_commerce"].task_count == 2
    assert rollup["internal_tools"].task_count == 1
    assert rollup["agentic_commerce"].total_cost_usd > rollup["internal_tools"].total_cost_usd


def test_tag_summary_tracks_escalation_rate_and_model_mix():
    tracker = TokenTracker()
    tags = {"initiative": "agentic_commerce"}
    tracker.record(make_task(TaskType.CODE_REVIEW, 100, 100, tags=tags, escalated=True))
    tracker.record(make_task(TaskType.CODE_REVIEW, 100, 100, tags=tags, escalated=False))

    summary = tracker.tag_summaries("initiative")[0]
    assert summary.escalation_rate == 0.5
    assert summary.models_used["mid-tier-model"] == 2


def test_tag_summaries_filter_by_key():
    tracker = TokenTracker()
    tracker.record(
        make_task(
            TaskType.CODE_REVIEW, 100, 100,
            tags={"initiative": "agentic_commerce", "env": "prod"},
        )
    )
    assert len(tracker.tag_summaries()) == 2
    assert len(tracker.tag_summaries("initiative")) == 1


def test_tag_budget_alert_flags_over_cap():
    tracker = TokenTracker()
    tracker.record(
        make_task(
            TaskType.CODE_REVIEW, 1_000_000, 1_000_000,
            tags={"initiative": "internal_tools"},
        )
    )
    governor = CostGovernor(
        tag_budgets=[TagBudget("initiative", "internal_tools", monthly_cap_usd=1.0)]
    )
    alerts = governor.check_tag_budgets(tracker)
    assert len(alerts) == 1
    assert alerts[0].is_over_cap is True


def test_tag_budget_without_matching_spend_produces_no_alert():
    tracker = TokenTracker()
    tracker.record(make_task(TaskType.CODE_REVIEW, 100, 100, tags={"initiative": "other"}))
    governor = CostGovernor(
        tag_budgets=[TagBudget("initiative", "not_present", monthly_cap_usd=1.0)]
    )
    assert governor.check_tag_budgets(tracker) == []
