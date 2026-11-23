
from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def append_or_reset(
    current: list[str] | None,
    update: list[str],
) -> list[str]:
    """
    Append new values during one graph run.

    ["__RESET__"] clears values from the previous turn.
    """
    if update == ["__RESET__"]:
        return []

    return (current or []) + (update or [])


class ResearchState(TypedDict, total=False):
    # Original question for the current turn.
    question: str

    # Conversation history.
    messages: Annotated[list[AnyMessage], add_messages]

    # Planning information.
    plan: list[str]
    current_task: str

    # Per-turn evidence and errors.
    tool_results: Annotated[list[str], append_or_reset]
    errors: Annotated[list[str], append_or_reset]

    # Execution control.
    iteration_count: int
    max_iterations: int
    status: str

    # Long-term memory.
    long_term_memories: list[str]
    memory_retrieval_status: str
    memory_write_status: str

    # Human-in-the-Loop.
    human_approved: bool | None
    approval_status: str

    # Final response.
    final_answer: str
