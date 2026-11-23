import pytest

from backend.services.tool_service import execute_tool


def test_execute_order_status_tool():
    result = execute_tool(
        "get_order_status",
        {"order_id": "ORD1001"},
    )

    assert "ORD1001" in result
    assert "shipped" in result


def test_unknown_tool():
    with pytest.raises(ValueError):
        execute_tool("delete_database", {})