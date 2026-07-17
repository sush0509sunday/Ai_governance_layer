import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_orchestrator import AgentOrchestrator, OrchestratorConfig
from src.models import ExecutionMode, TaskType


def test_high_confidence_task_routes_to_governed_agent():
    orchestrator = AgentOrchestrator()
    task = orchestrator.process("T1", "please review this pull request before merge")
    assert task.execution_mode == ExecutionMode.GOVERNED_AGENT
    assert task.token_usage is not None


def test_low_confidence_task_falls_back_to_manual():
    orchestrator = AgentOrchestrator(config=OrchestratorConfig(classification_confidence_threshold=0.99))
    task = orchestrator.process("T1", "please review this pull request before merge")
    assert task.execution_mode == ExecutionMode.UNGOVERNED_MANUAL
    assert task.token_usage is None


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
