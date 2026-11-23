def evaluate_tool_usage(
    state: dict,
    expected_tool: str | None,
    requires_human_approval: bool,
) -> dict:
    """
    Evaluate tool selection, execution, and human approval
    using the final or interrupted LangGraph state.
    """

    messages = state.get("messages", [])
    tool_results = state.get("tool_results", [])
    errors = state.get("errors", [])

    selected_tools = []
    executed_tools = []

    # -------------------------------------------------
    # 1. Extract selected tools from AI messages
    # -------------------------------------------------

    for message in messages:
        tool_calls = getattr(message, "tool_calls", [])

        for call in tool_calls:
            tool_name = call.get("name")

            if tool_name:
                selected_tools.append(tool_name)

    # -------------------------------------------------
    # 2. Extract executed tools from ToolMessages
    # -------------------------------------------------

    for message in messages:
        tool_call_id = getattr(message, "tool_call_id", None)

        if tool_call_id:
            tool_name = getattr(message, "name", None)

            if tool_name:
                executed_tools.append(tool_name)

    # -------------------------------------------------
    # 3. Evaluate tool selection
    # -------------------------------------------------

    if expected_tool is None:
        selection_passed = len(selected_tools) == 0
    else:
        selection_passed = expected_tool in selected_tools

    # -------------------------------------------------
    # 4. Evaluate tool execution
    # -------------------------------------------------

    if expected_tool is None:
        execution_status = "not_required"
        execution_passed = True

    elif expected_tool not in executed_tools:
        execution_status = "not_executed"
        execution_passed = False

    else:
        matching_results = [
            str(result)
            for result in tool_results
            if str(result).startswith(f"{expected_tool}:")
        ]

        failure_markers = (
            "WEB_SEARCH_FAILED",
            "Calculation error:",
            "RAG_SEARCH_FAILED",
            "NO_RELEVANT_DOCUMENTS",
            "No web search results",
        )

        has_error = any(
            marker in result
            for result in matching_results
            for marker in failure_markers
        )

        execution_status = (
            "failed" if has_error else "executed"
        )

        execution_passed = not has_error

    # -------------------------------------------------
    # 5. Evaluate human approval
    # -------------------------------------------------

    approval_status = state.get(
        "approval_status",
        "not_required",
    )

    graph_status = state.get("status", "unknown")

    if not requires_human_approval:
        approval_passed = True

    elif (
        graph_status == "tool_requested"
        or approval_status in ("approved", "rejected")
    ):
        approval_passed = True

    else:
        approval_passed = False

    if graph_status == "tool_requested":
        approval_result = "pending_approval"
    elif approval_status == "approved":
        approval_result = "approved"
    elif approval_status == "rejected":
        approval_result = "rejected"
    elif not requires_human_approval:
        approval_result = "not_required"
    else:
        approval_result = "not_requested"

    # -------------------------------------------------
    # 6. Final evaluation
    # -------------------------------------------------

    return {
        "expected_tool": expected_tool,
        "selected_tools": selected_tools,
        "executed_tools": executed_tools,
        "selection_passed": selection_passed,
        "execution_passed": execution_passed,
        "execution_status": execution_status,
        "approval_passed": approval_passed,
        "approval_status": approval_result,
        "overall_passed": (
            selection_passed
            and execution_passed
            and approval_passed
        ),
    }