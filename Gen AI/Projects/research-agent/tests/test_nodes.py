
from types import SimpleNamespace

from app.agent import nodes


class FakeToolExecutor:
    """Simulate a tool executor without making real tool calls."""

    def __init__(self, messages):
        self.messages = messages

    def invoke(self, state, config=None):
        return {"messages": self.messages}


def test_web_search_error_is_recorded(monkeypatch):
    """A failed search should appear in the errors list."""

    fake_message = SimpleNamespace(
        name="web_search",
        content=(
            "WEB_SEARCH_FAILED: External provider unavailable."
        ),
    )

    monkeypatch.setattr(
        nodes,
        "tool_executor",
        FakeToolExecutor([fake_message]),
    )

    result = nodes.tools_node(
        {"messages": []},
        config={},
    )

    assert result["status"] == "tool_error"
    assert len(result["tool_results"]) == 1
    assert len(result["errors"]) == 1
    assert "WEB_SEARCH_FAILED" in result["errors"][0]


def test_successful_tool_has_no_errors(monkeypatch):
    """A successful calculation should not be recorded as an error."""

    fake_message = SimpleNamespace(
        name="calculator",
        content="360",
    )

    monkeypatch.setattr(
        nodes,
        "tool_executor",
        FakeToolExecutor([fake_message]),
    )

    result = nodes.tools_node(
        {"messages": []},
        config={},
    )

    assert result["status"] == "tool_completed"
    assert result["tool_results"] == ["calculator: 360"]
    assert result["errors"] == []


def test_no_search_results_are_recorded_as_error(monkeypatch):
    """An empty search result should be visible for recovery."""

    fake_message = SimpleNamespace(
        name="web_search",
        content="No web search results were found.",
    )

    monkeypatch.setattr(
        nodes,
        "tool_executor",
        FakeToolExecutor([fake_message]),
    )

    result = nodes.tools_node(
        {"messages": []},
        config={},
    )

    assert result["status"] == "tool_error"
    assert len(result["errors"]) == 1
