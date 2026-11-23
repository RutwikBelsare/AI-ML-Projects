
from app.agent.state import ResearchState
import logging

logger = logging.getLogger(__name__)

def route_after_agent(state: ResearchState) -> str:
    """Route based on tool calls and the iteration budget."""

    messages = state.get("messages", [])

    if not messages:
        return "finalize"

    last_message = messages[-1]

    has_tool_calls = bool(
        getattr(last_message, "tool_calls", [])
    )

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    if has_tool_calls and iteration_count < max_iterations:
        return "human_review"

    return "finalize"


def route_after_approval(state: ResearchState) -> str:
    """
    Route according to the human approval decision.
    """

    approved = state.get("human_approved")

    if not isinstance(approved, bool):
        raise ValueError(
            "Invalid approval state: human_approved must be a boolean."
        )

    logger.info(
        "ROUTE AFTER APPROVAL | human_approved=%s",
        approved,
    )

    if approved:
        logger.info("Routing to TOOLS node")
        return "tools"

    logger.info("Routing to REJECT_TOOL_CALLS node")
    return "reject_tool_calls"