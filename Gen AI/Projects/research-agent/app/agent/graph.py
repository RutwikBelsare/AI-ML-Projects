from pathlib import Path
import sqlite3
import atexit
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from app.logging_config import configure_logging
from langchain_core.messages import AIMessageChunk # used for streaming process

from app.agent.state import ResearchState
from app.agent.nodes import (
    planner_node,
    retrieve_memories_node,
    agent_node,
    human_review_node,
    reject_tool_calls_node,
    tools_node,
    finalize_node,
    is_personal_memory_question,
)
from app.memory.store import long_term_store
from app.agent.edges import (route_after_agent, route_after_approval)
from app.rag.retriever import search_documents



logger = configure_logging(__name__)

# RUNTIME INPUT VALIDATION
MAX_QUESTION_LENGTH = 5000
MAX_IDENTIFIER_LENGTH = 128


def _validate_text(value: str, field_name: str, max_length: int) -> str:
    """Validate and normalize a required text input."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty or whitespace.")

    if len(normalized) > max_length:
        raise ValueError(
            f"{field_name} cannot exceed {max_length} characters."
        )

    return normalized


def _validate_request(question: str, thread_id: str, user_id: str):
    """Validate the required inputs for a new graph run."""
    return (
        _validate_text(question, "Question", MAX_QUESTION_LENGTH),
        _validate_text(thread_id, "Thread ID", MAX_IDENTIFIER_LENGTH),
        _validate_text(user_id, "User ID", MAX_IDENTIFIER_LENGTH),
    )


def _validate_resume_inputs(thread_id: str, user_id: str, approved: bool):
    """Validate identifiers and the human approval decision."""
    thread_id = _validate_text(
        thread_id, "Thread ID", MAX_IDENTIFIER_LENGTH
    )
    user_id = _validate_text(
        user_id, "User ID", MAX_IDENTIFIER_LENGTH
    )

    if not isinstance(approved, bool):
        raise TypeError("Approval must be True or False.")

    return thread_id, user_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents = True, exist_ok = True)
CHECKPOINT_DB_PATH = CHECKPOINT_DIR /  "research_agent.sqlite"

sqlite_connection = sqlite3.connect(
    str(CHECKPOINT_DB_PATH),
    check_same_thread = False
)


checkpointer = SqliteSaver(sqlite_connection)
checkpointer.setup()
atexit.register(sqlite_connection.close)


#1. Dedicated RAG retrieval node

def retrieve_documents_node(state: ResearchState) -> dict:
    question = state.get("question", "").strip()

    try:
        matches = search_documents(
            query=question,
            k=4,
        )

        MAX_RETRIEVAL_DISTANCE = 1.1

        relevant_matches = [
            match
            for match in matches
            if match.get("distance", float("inf"))
            <= MAX_RETRIEVAL_DISTANCE
        ]

        # Log all retrieved chunks.
        logger.info(
            "RAG RETRIEVAL | query=%s | total_matches=%d",
            question,
            len(matches),
        )

        for match in matches:
            logger.info(
                "RAG MATCH | source=%s | distance=%s | accepted=%s",
                match.get("source"),
                match.get("distance"),
                match.get("distance", float("inf"))
                <= MAX_RETRIEVAL_DISTANCE,
            )

        if not relevant_matches:
            result = (
                "NO_RELEVANT_DOCUMENTS: No document chunks "
                "passed the retrieval distance threshold."
            )

            return {
                "tool_results": [
                    f"document_search: {result}"
                ],
                "errors": [result],
                "status": "retrieval_no_match",
            }

        formatted_results = []

        for match in relevant_matches:

            formatted_results.append(
                f"Source: {match.get('source', 'unknown')}\n"
                f"Distance: {match.get('distance')}\n"
                f"Content:\n{match['content']}"
            )

        result = "\n\n--- Retrieved passage ---\n\n".join(
            formatted_results
        )

        return {
            "tool_results": [
                f"document_search:\n{result}"
            ],
            "status": "retrieval_complete",
        }

    except Exception as exc:

        logger.exception("RAG document retrieval failed")

        error = (
            f"RAG_SEARCH_FAILED: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "tool_results": [
                f"document_search: {error}"
            ],
            "errors": [error],
            "status": "retrieval_failed",
        }


# 2.Routing after Retrieval




def route_after_retrieval(state: ResearchState) -> str:
    """
    Route retrieved context to the agent.

    The agent decides whether to answer directly
    or request an external tool.
    """

    status = state.get("status")

    logger.info(
        "ROUTE AFTER RETRIEVAL | status=%s | destination=agent",
        status,
    )

    return "agent"


def route_after_planner(state: ResearchState) -> str:
    """
    Route personal-memory requests directly to finalization.
    Other questions continue through RAG.
    """

    question = state.get("question", "")

    if is_personal_memory_question(question):
        return "finalize"

    return "retrieve"

# 3. Build the Graph

builder = StateGraph(ResearchState)

# Register nodes
builder.add_node("memory_retrieve", retrieve_memories_node)
builder.add_node("planner", planner_node)
builder.add_node("retrieve", retrieve_documents_node)
builder.add_node("agent", agent_node)

# Human-in-the-Loop nodes
builder.add_node("human_review", human_review_node)
builder.add_node("reject_tool_calls", reject_tool_calls_node)

builder.add_node("tools", tools_node)
builder.add_node("finalize", finalize_node)


# Initial workflow
builder.add_edge(START, "memory_retrieve")
builder.add_edge("memory_retrieve", "planner")
builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "finalize": "finalize",
        "retrieve": "retrieve",
    },
)


# Route after RAG retrieval
builder.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "agent": "agent",
        "finalize": "finalize",
    },
)


# Route after agent decides whether tools are needed
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "human_review": "human_review",
        "finalize": "finalize",
    },
)


# Route after human approval
builder.add_conditional_edges(
    "human_review",
    route_after_approval,
    {
        "tools": "tools",
        "reject_tool_calls": "reject_tool_calls",
    },
)


# Tool execution returns to agent; rejection terminates workflow immediately
builder.add_edge("tools", "agent")
builder.add_edge("reject_tool_calls", END)

# Final response
builder.add_edge("finalize", END)


# Compile graph with persistent checkpointing and long-term memory
research_graph = builder.compile(
    checkpointer=checkpointer,
    store=long_term_store,
)



# CENTRAL GRAPH EXCEPTION HANDLING
def _invoke_graph_safely(
    graph_input,
    config: dict,
    operation: str,
) -> dict:
    """Invoke the graph and handle unexpected execution failures."""

    try:
        return research_graph.invoke(
            graph_input,
            config=config,
        )

    except Exception as exc:
        logger.exception(
            "Graph execution failed | operation=%s",
            operation,
        )

        raise RuntimeError(
            "Research workflow failed. "
            "Check logs/research_agent.log for details."
        ) from exc



def _stream_graph_safely(
    graph_input,
    config: dict,
):
    """Stream graph events and handle failures during iteration."""

    try:
        yield from research_graph.stream(
            graph_input,
            config=config,
            stream_mode=["updates", "messages"],
        )

    except Exception as exc:
        logger.exception("Streaming graph execution failed")

        raise RuntimeError(
            "Research streaming failed. "
            "Check logs/research_agent.log for details."
        ) from exc



# 4. Run the research workflow
def run_research(
    question: str,
    thread_id: str = "default",
    user_id: str = "local_user",
) -> dict:
    """Run the workflow and return the final state."""

    question, thread_id, user_id = _validate_request(
        question, thread_id, user_id
    )

    logger.info(
        "Research execution started | thread_id=%s | user_id=%s",
        thread_id,
        user_id,
    )


    return _invoke_graph_safely(
    graph_input={
        "question": question,
        "messages": [HumanMessage(content=question)],
        "tool_results": ["__RESET__"],
        "errors": ["__RESET__"],
        "iteration_count": 0,
        "max_iterations": 5,
        "status": "started",
        "human_approved": None,
        "approval_status": "not_required",
    },
    config={
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
        "recursion_limit": 30,
    },
    operation="run_research",
)




def stream_graph_execution(
    graph_input,
    config: dict,
    execution_trace: list[str] | None = None,
    event_callback=None,
) -> tuple[dict, list]:
    """
    Stream graph node updates and final LLM response tokens.

    Supports:
    - updates: Node-level state updates.
    - messages: LLM message chunks.
    - HITL: Human approval interrupts.
    """
    logger.info("Streaming graph execution started")
    pending_interrupts = []
    final_answer_started = False

    for event in _stream_graph_safely(
        graph_input,
        config,
    ):

        # Handle multi-mode events.
        if isinstance(event, tuple) and len(event) == 2:
            mode, data = event

        # Also support the v2 event structure if used later.
        elif isinstance(event, dict) and "type" in event:
            mode = event["type"]
            data = event.get("data")

        else:
            continue


        # 1. Handle LLM message chunks
        if mode == "messages":

            if not isinstance(data, tuple) or len(data) != 2:
                continue

            message_chunk, metadata = data

            # Stream only incremental chunks from the finalize node.
            # This skips planner/agent output and the final accumulated AIMessage.
            if metadata.get("langgraph_node") != "finalize":
                continue

            if not isinstance(message_chunk, AIMessageChunk):
                continue

            content = getattr(message_chunk, "content", "")

            # Extract text from string or content blocks.
            if isinstance(content, str):
                text_chunk = content

            elif isinstance(content, list):
                text_chunk = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                )

            else:
                text_chunk = ""

            if text_chunk:
                if not final_answer_started:
                    print(
                        "\n========== FINAL ANSWER (STREAMING) ==========\n"
                    )
                    final_answer_started = True

                print(text_chunk, end="", flush=True)
                if event_callback is not None:
                    event_callback({
                        "type": "token",
                        "content": text_chunk,
                    })

            continue


        # 2. Handle node-level updates
        if mode == "updates":

            if not isinstance(data, dict):
                continue

            # Handle HITL interrupts.
            if "__interrupt__" in data:

                pending_interrupts.extend(
                    data["__interrupt__"]
                )

                if event_callback is not None:
                    for interrupt_item in data["__interrupt__"]:
                        event_callback({
                            "type": "approval_required",
                            "content": str(interrupt_item.value),
                        })

                for interrupt_item in data["__interrupt__"]:
                    print(
                        "\n========== HUMAN APPROVAL REQUIRED =========="
                    )
                    print(interrupt_item.value)

                continue

            for node_name, node_update in data.items():
                logger.info("Graph node executed | node=%s", node_name)

                if event_callback is not None:
                    event_callback({
                        "type": "node_update",
                        "node": node_name,
                        "status": (
                            node_update.get("status")
                            if isinstance(node_update, dict)
                            else None
                        ),
                    })

                if execution_trace is not None:
                    execution_trace.append(node_name)

                print(
                    f"\n========== NODE: {node_name.upper()} =========="
                )
                

                if not isinstance(node_update, dict):
                    print(node_update)
                    continue

                if node_update.get("status"):
                    print("Status:", node_update["status"])

                if node_update.get("plan"):
                    print("Research plan:")

                    for index, task in enumerate(
                        node_update["plan"],
                        start=1,
                    ):
                        print(f"{index}. {task}")

                if node_update.get("tool_results"):
                    print("Tool results:")

                    for result in node_update["tool_results"]:
                        print(result)


    # 3. Read the latest persisted state

    snapshot = research_graph.get_state(config)

    result = snapshot.values

    # Fallback in case the model/provider did not emit
    # visible message chunks.
    # Do not display a previous answer when the graph
# is paused and waiting for human approval.
    if not final_answer_started and not pending_interrupts:

        final_answer = result.get("final_answer")

        if final_answer:
            print("\n========== FINAL ANSWER ==========\n")
            print(final_answer)

    logger.info(
    "Streaming graph execution finished | status=%s | pending_interrupts=%s",
    result.get("status", "unknown"),
    len(pending_interrupts),
)

    if event_callback is not None:
        event_callback({
            "type": "completed",
            "status": result.get("status", "unknown"),
            "final_answer": result.get("final_answer", ""),
            "approval_required": bool(pending_interrupts),
            "errors": result.get("errors", []),
        })

    return result, pending_interrupts





def run_research_streaming(
    question: str,
    thread_id: str = "default",
    user_id: str = "local_user",
    execution_trace: list[str] | None = None,
    event_callback=None,
) -> tuple[dict, list]:
    """Start a research workflow with real-time node updates."""

    question, thread_id, user_id = _validate_request(
        question, thread_id, user_id
    )

    config = {
        "configurable": {
            "thread_id": thread_id.strip(),
            "user_id": user_id.strip(),
        },
        "recursion_limit": 30,
    }

    initial_state = {
    "question": question.strip(),
    "messages": [HumanMessage(content=question.strip())],
    "tool_results": ["__RESET__"],
    "errors": ["__RESET__"],
    "iteration_count": 0,
    "max_iterations": 5,
    "status": "started",
    "human_approved": None,
    "approval_status": "not_required",
    "final_answer": "",
    }

    return stream_graph_execution(
        initial_state,
        config,
        execution_trace=execution_trace,
        event_callback=event_callback,
    )


def resume_research_streaming(
    thread_id: str,
    user_id: str,
    approved: bool,
) -> tuple[dict, list]:
    """Resume an interrupted workflow with streaming updates."""

    thread_id, user_id = _validate_resume_inputs(
        thread_id, user_id, approved
    )

    logger.info(
        "Resuming interrupted workflow | thread_id=%s | approved=%s",
        thread_id,
        approved,
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
        "recursion_limit": 30,
    }

    return stream_graph_execution(
        Command(resume={"approved": approved}),
        config,
    )
  



# Resume interrupted research workflow
# Resume interrupted research workflow
def resume_research(
    thread_id: str,
    user_id: str,
    approved: bool,
) -> dict:
    """
    Resume an interrupted LangGraph workflow
    after human approval or rejection.
    """

    thread_id, user_id = _validate_resume_inputs(
        thread_id,
        user_id,
        approved,
    )

    logger.info(
        "RESUME REQUEST RECEIVED | thread_id=%s | user_id=%s | approved=%s",
        thread_id,
        user_id,
        approved,
    )
    logger.info(
    "APPROVAL DEBUG BEFORE RESUME | "
    "thread_id=%s | approved=%s",
    thread_id,
    approved,
)

    result = _invoke_graph_safely(
        graph_input=Command(
            resume={
                "approved": approved,
            }
        ),
        config={
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            "recursion_limit": 30,
        },
        operation="resume_research",
    )

    logger.info(
        "RESUME COMPLETED | thread_id=%s | status=%s | approval_status=%s",
        thread_id,
        result.get("status"),
        result.get("approval_status"),
    )

    return result

# 5. Command-line entry point



if __name__ == "__main__":
    question = input("Enter your research question: ").strip()

    thread_id = (
        input("Enter conversation thread_id: ").strip()
        or "default"
    )

    user_id = (
        input("Enter user_id: ").strip()
        or "local_user"
    )

    try:
        # Start graph with streaming.
        result, pending_interrupts = run_research_streaming(
            question=question,
            thread_id=thread_id,
            user_id=user_id,
        )

        # Handle human approval requests.
        while pending_interrupts:
            decision = input(
                "\nApprove these tool calls? (yes/no): "
            ).strip().lower()

            if decision not in ("yes", "no"):
                print("Please enter yes or no.")
                continue

            result, pending_interrupts = resume_research_streaming(
                thread_id=thread_id,
                user_id=user_id,
                approved=(decision == "yes"),
            )

        print("\n========== FINAL EXECUTION SUMMARY ==========")

        print("Status:", result.get("status", "unknown"))
        print("Approval:", result.get("approval_status", "unknown"))
        print("Memory write:", result.get("memory_write_status", "unknown"))
        print("Agent iterations:", result.get("iteration_count", 0))

        final_answer = result.get("final_answer", "")
        print("Final answer saved:", bool(final_answer))
        print("Final answer length:", len(final_answer) if final_answer else 0)

        if result.get("errors"):
            print("\nErrors:")
            for error in result["errors"]:
                print("-", error)

    except Exception as exc:
        print(f"\nResearch failed: {type(exc).__name__}: {exc}")
        raise
