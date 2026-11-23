
from langchain_core.messages import AIMessage

from app.agent.edges import route_after_agent


def test_final_answer_routes_to_finalize():
    """A normal answer should execute the finalization node."""

    state = {
        "messages": [
            AIMessage(content="LangGraph helps orchestrate workflows.")
        ],
        "iteration_count": 1,
        "max_iterations": 5,
    }

    assert route_after_agent(state) == "finalize"


def test_tool_call_routes_to_tools():
    """A requested tool should route to the tools node."""

    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "15 * 24"},
                "id": "call_test_1",
                "type": "tool_call",
            }
        ],
    )

    state = {
        "messages": [message],
        "iteration_count": 1,
        "max_iterations": 5,
    }

    assert route_after_agent(state) == "tools"


def test_iteration_limit_routes_to_finalize():
    """A tool request at the limit should trigger finalization."""

    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "LangGraph documentation"},
                "id": "call_test_2",
                "type": "tool_call",
            }
        ],
    )

    state = {
        "messages": [message],
        "iteration_count": 5,
        "max_iterations": 5,
    }

    assert route_after_agent(state) == "finalize"


def test_empty_messages_routes_to_finalize():
    """An empty message history should route safely to finalization."""

    assert route_after_agent({"messages": []}) == "finalize"
