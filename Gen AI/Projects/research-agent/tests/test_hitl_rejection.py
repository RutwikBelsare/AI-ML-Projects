from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.agent.nodes import reject_tool_calls_node, finalize_node
from app.agent.edges import route_after_approval
from app.agent.graph import research_graph


def test_reject_tool_calls_node_cancels_execution():
    """Rejecting tool calls must set status to rejected and provide a cancellation answer."""
    state = {
        "question": "Calculate 4000 * 40",
        "messages": [
            HumanMessage(content="Calculate 4000 * 40"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "4000 * 40"},
                        "id": "call_calc_999",
                    }
                ],
            ),
        ],
    }

    result = reject_tool_calls_node(state)

    assert result["status"] == "rejected"
    assert result["approval_status"] == "rejected"
    assert result["human_approved"] is False
    assert "cancelled" in result["final_answer"].lower()
    assert "calculator" in result["final_answer"].lower()

    # Rejection messages must include ToolMessage resolving the call and an AIMessage
    messages = result["messages"]
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "call_calc_999" for m in messages)
    assert any(isinstance(m, AIMessage) and "cancelled" in m.content.lower() for m in messages)


def test_finalize_node_safeguard_on_rejection():
    """finalize_node must return cancellation text without invoking LLM when rejected."""
    mock_store = MagicMock()
    state = {
        "question": "Calculate 4000 * 40",
        "human_approved": False,
        "approval_status": "rejected",
        "final_answer": "Execution cancelled: The human reviewer rejected calculator.",
    }

    config = {"configurable": {"user_id": "test_user"}}
    result = finalize_node(state, config=config, store=mock_store)

    assert result["status"] == "rejected"
    assert result["approval_status"] == "rejected"
    assert "cancelled" in result["final_answer"].lower()


def test_route_after_approval_rejection():
    """route_after_approval must route to reject_tool_calls when human_approved is False."""
    assert route_after_approval({"human_approved": False}) == "reject_tool_calls"
    assert route_after_approval({"human_approved": True}) == "tools"


def test_reject_tool_calls_routes_to_end():
    """reject_tool_calls must terminate at END, never loop back to agent."""
    graph = research_graph.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    # Rejection terminates at __end__
    assert ("reject_tool_calls", "__end__") in edges

    # Rejection must NOT loop back to agent
    assert ("reject_tool_calls", "agent") not in edges
