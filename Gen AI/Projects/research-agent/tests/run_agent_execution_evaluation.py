from uuid import uuid4

from app.agent.graph import run_research_streaming
from tests.agent_execution_evaluation import (
    evaluate_agent_execution,
)


def run_execution_evaluation():

    print("\n========== REAL AGENT EXECUTION EVALUATION ==========\n")

    question = "What is LangGraph?"

    expected_nodes = [
    "memory_retrieve",
    "planner",
    "retrieve",
    "finalize",
]

    execution_trace = []

    thread_id = f"execution_eval_{uuid4().hex[:8]}"

    state, pending_interrupts = run_research_streaming(
        question=question,
        thread_id=thread_id,
        user_id="evaluation_user",
        execution_trace=execution_trace,
    )

    # Pass actual observed nodes to the evaluator.
    evaluation_state = {
        **state,
        "executed_nodes": execution_trace,
    }

    result = evaluate_agent_execution(
        state=evaluation_state,
        expected_nodes=expected_nodes,
        max_iterations=5,
        pending_interrupts=pending_interrupts,
    )

    print("\n========== EXECUTION RESULTS ==========")

    print("Expected nodes:", expected_nodes)
    print("Actual nodes:", execution_trace)

    print("Missing nodes:", result["missing_nodes"])

    print("Iteration count:", result["iteration_count"])
    print("Workflow status:", result["workflow_status"])

    print("Workflow completed:", result["workflow_completed"])
    print("Execution errors:", state.get("errors", []))

    print("Node execution passed:", result["node_execution_passed"])
    print("Iteration check passed:", result["iteration_passed"])

    print("\nOverall evaluator result:", result["overall_passed"])

    # This particular test expects a completed direct-answer workflow.
    if not result["workflow_completed"]:
        raise AssertionError("Expected workflow to complete.")

    if not result["overall_passed"]:
        raise AssertionError("Agent execution evaluation failed.")

    print("\nREAL EXECUTION EVALUATION PASSED")

    return result


if __name__ == "__main__":
    run_execution_evaluation()