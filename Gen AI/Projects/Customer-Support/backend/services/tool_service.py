import json

from backend.tools.order_tools import (
    get_order_status,
)


AVAILABLE_FUNCTIONS = {
    "get_order_status": get_order_status,
}


def execute_tool(
    tool_name: str,
    arguments: dict,
) -> str:

    function = AVAILABLE_FUNCTIONS.get(
        tool_name
    )

    if function is None:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    return function(**arguments)