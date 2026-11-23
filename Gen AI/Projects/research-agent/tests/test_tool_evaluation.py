from langchain_core.messages import AIMessage, ToolMessage

from tests.tool_evaluation import evaluate_tool_usage


def test_correct_tool_selection_and_execution():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "15 * 24"},
                        "id": "call_123",
                    }
                ],
            ),
            ToolMessage(
                content="360",
                tool_call_id="call_123",
                name="calculator",
            ),
        ],
        "tool_results": ["calculator: 360"],
        "errors": [],
        "status": "completed",
        "approval_status": "approved",
    }

    result = evaluate_tool_usage(
        state,
        expected_tool="calculator",
        requires_human_approval=True,
    )

    assert result["selection_passed"] is True
    assert result["execution_passed"] is True
    assert result["approval_passed"] is True
    assert result["overall_passed"] is True


def test_correct_selection_but_pending_approval():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "15 * 24"},
                        "id": "call_456",
                    }
                ],
            )
        ],
        "tool_results": [],
        "errors": [],
        "status": "tool_requested",
        "approval_status": "not_required",
    }

    result = evaluate_tool_usage(
        state,
        expected_tool="calculator",
        requires_human_approval=True,
    )

    assert result["selection_passed"] is True
    assert result["execution_status"] == "not_executed"
    assert result["approval_passed"] is True
    assert result["approval_status"] == "pending_approval"
    assert result["overall_passed"] is False


def test_direct_answer_without_tool():
    state = {
        "messages": [
            AIMessage(content="LangGraph is a workflow framework.")
        ],
        "tool_results": [],
        "errors": [],
        "status": "completed",
        "approval_status": "not_required",
    }

    result = evaluate_tool_usage(
        state,
        expected_tool=None,
        requires_human_approval=False,
    )

    assert result["selection_passed"] is True
    assert result["execution_passed"] is True
    assert result["overall_passed"] is True


def test_tool_execution_failure():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "LangGraph"},
                        "id": "call_789",
                    }
                ],
            ),
            ToolMessage(
                content="WEB_SEARCH_FAILED: connection timeout",
                tool_call_id="call_789",
                name="web_search",
            ),
        ],
        "tool_results": [
            "web_search: WEB_SEARCH_FAILED: connection timeout"
        ],
        "errors": ["web_search: connection timeout"],
        "status": "tool_error",
        "approval_status": "approved",
    }

    result = evaluate_tool_usage(
        state,
        expected_tool="web_search",
        requires_human_approval=True,
    )

    assert result["selection_passed"] is True
    assert result["execution_passed"] is False
    assert result["execution_status"] == "failed"
    assert result["overall_passed"] is False