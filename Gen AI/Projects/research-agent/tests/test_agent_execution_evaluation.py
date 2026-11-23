from tests.agent_execution_evaluation import (
    evaluate_agent_execution,
)


def test_successful_workflow_execution():
    state = {
        "executed_nodes": [
            "memory_retrieve",
            "planner",
            "retrieve",
            "agent",
            "finalize",
        ],
        "iteration_count": 1,
        "status": "completed",
        "approval_status": "not_required",
        "errors": [],
    }

    result = evaluate_agent_execution(
        state=state,
        expected_nodes=[
            "memory_retrieve",
            "planner",
            "retrieve",
            "agent",
            "finalize",
        ],
    )

    assert result["node_execution_passed"] is True
    assert result["iteration_passed"] is True
    assert result["workflow_completed"] is True
    assert result["overall_passed"] is True


def test_iteration_limit():
    state = {
        "executed_nodes": ["agent", "finalize"],
        "iteration_count": 6,
        "status": "completed",
        "errors": [],
    }

    result = evaluate_agent_execution(
        state=state,
        expected_nodes=["agent", "finalize"],
        max_iterations=5,
    )

    assert result["iteration_passed"] is False
    assert result["overall_passed"] is False


def test_pending_human_approval():
    state = {
        "executed_nodes": [
            "memory_retrieve",
            "planner",
            "retrieve",
            "agent",
        ],
        "iteration_count": 1,
        "status": "tool_requested",
        "approval_status": "not_required",
        "errors": [],
    }

    result = evaluate_agent_execution(
        state=state,
        expected_nodes=[
            "memory_retrieve",
            "planner",
            "retrieve",
            "agent",
        ],
        pending_interrupts=[{"type": "human_approval_required"}],
    )

    assert result["approval_pending"] is True
    assert result["approval_status"] == "pending"
    assert result["workflow_completed"] is False
    assert result["overall_passed"] is True