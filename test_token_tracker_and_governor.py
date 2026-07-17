import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cost_governor import Budget, CostGovernor
from src.models import ExecutionMode, GovernedTask, PriceTable, TaskType, TokenUsage
from src.token_tracker import TokenTracker


def make_task(task_type: TaskType, input_tokens: int, output_tokens: int, model="mid-tier-model") -> GovernedTask:
    return GovernedTask(
        task_id="T1",
        raw_text="dummy",
        task_type=task_type,
        execution_mode=ExecutionMode.GOVERNED_AGENT,
        token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, model_name=model),
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


def test_budget_alert_flags_over_cap():
    tracker = TokenTracker()
    tracker.record(make_task(TaskType.CODE_REVIEW, 1_000_000, 1_000_000))  # deliberately huge

    governor = CostGovernor(budgets=[Budget(TaskType.CODE_REVIEW, monthly_cap_usd=1.0)])
    alerts = governor.check_budgets(tracker)

    assert len(alerts) == 1
    assert alerts[0].is_over_cap is True


def test_estimate_avoided_cost_is_nonnegative():
    tracker = TokenTracker()
    tracker.record(make_task(TaskType.SUPPORT_QUERY, 100, 200))
    summary = tracker.summaries()[0]
    avoided = CostGovernor.estimate_avoided_cost(summary)
    assert avoided >= 0
