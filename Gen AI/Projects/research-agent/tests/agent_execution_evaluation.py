def evaluate_agent_execution(
    state: dict,
    expected_nodes: list[str],
    max_iterations: int = 5,
    pending_interrupts: list | None = None,
) -> dict:
    """
    Evaluate the final state and observed graph execution.

    Checks workflow completion, expected nodes,
    iteration limits, and human approval interrupts.
    """

    pending_interrupts = pending_interrupts or []

    actual_nodes = state.get("executed_nodes", [])

    # 1. Check expected node execution.
    missing_nodes = [
        node
        for node in expected_nodes
        if node not in actual_nodes
    ]

    node_execution_passed = len(missing_nodes) == 0

    # 2. Check iteration limit.
    iteration_count = state.get("iteration_count", 0)

    iteration_passed = (
        isinstance(iteration_count, int)
        and iteration_count <= max_iterations
    )

    # 3. Check workflow status.
    status = state.get("status", "unknown")

    workflow_completed = (
        status == "completed"
        and not pending_interrupts
    )

    # 4. Check approval interrupt.
    approval_pending = len(pending_interrupts) > 0

    if approval_pending:
        approval_status = "pending"
    else:
        approval_status = state.get(
            "approval_status",
            "not_required",
        )

    # 5. Check execution errors.
    errors = state.get("errors", [])

    no_execution_errors = len(errors) == 0

    return {
        "actual_nodes": actual_nodes,
        "expected_nodes": expected_nodes,
        "missing_nodes": missing_nodes,
        "node_execution_passed": node_execution_passed,
        "iteration_count": iteration_count,
        "iteration_passed": iteration_passed,
        "workflow_status": status,
        "workflow_completed": workflow_completed,
        "approval_status": approval_status,
        "approval_pending": approval_pending,
        "no_execution_errors": no_execution_errors,
        "overall_passed": (
            node_execution_passed
            and iteration_passed
            and no_execution_errors
        ),
    }