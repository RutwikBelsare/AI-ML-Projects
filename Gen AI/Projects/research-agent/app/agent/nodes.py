
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from app.logging_config import configure_logging
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from uuid import uuid4
import re
from langgraph.prebuilt import ToolNode
from app.llm import llm
from app.tools.research_tools import tools
from app.agent.state import ResearchState
from app.memory.store import get_user_memory_namespace

# 1. Bind the complete registered tool list to the LLM.
#
# Keep this binding global and stable. Dynamically removing tools between
# requests previously caused Groq to report calls to tools that were not
# included in the current request.
model_with_tools = llm.bind_tools(tools)
tool_executor = ToolNode(tools)
logger = configure_logging(__name__)



# LONG-TERM MEMORY CRUD HELPERS

def _memory_matches_subject(memory_text: str, subject: str) -> bool:
    """Check whether a stored memory describes the requested subject."""
    ignored = {"my", "the", "a", "an", "is", "are", "of", "about"}

    subject_words = {
        word for word in re.findall(r"[a-z0-9]+", subject.lower())
        if word not in ignored
    }
    memory_words = set(re.findall(r"[a-z0-9]+", memory_text.lower()))

    return bool(subject_words) and subject_words.issubset(memory_words)



def _find_matching_memories(store, namespace, subject: str):
    """Find stored memories matching a subject."""
    items = store.search(namespace, query=subject, limit=20)

    return [
        item for item in items
        if _memory_matches_subject(
            str((item.value or {}).get("text", "")),
            subject,
        )
    ]




def _handle_memory_request(question: str, store, user_id: str):
    """
    Handle explicit remember, update, and forget requests.

    Returns a result dictionary, or None if the request is
    not a recognized memory operation.
    """
    namespace = get_user_memory_namespace(user_id)
    text = question.strip().rstrip(".!?")
    lower_text = text.lower()

    # UPDATE: Change an existing memory.
    update_match = re.match(
        r"(?:(?:please\s+)?(?:update|change|replace)\s+|"
        r"(?:i\s+)?(?:have\s+)?(?:switched|changed)\s+)"
        r"(.+?)\s*to\s+(.+)$",
        text,
        re.IGNORECASE,
    )

    if update_match:
        subject = update_match.group(1).strip()
        new_value = update_match.group(2).strip()
        matches = _find_matching_memories(store, namespace, subject)

        if not matches:
            return {
                "answer": (
                    f"I couldn't find an existing memory about "
                    f"'{subject}'. Nothing was changed."
                ),
                "status": "not_found",
            }

        new_memory_text = f"{subject} is {new_value}"

        for item in matches:
            category = (item.value or {}).get("category", "user_memory")
            store.put(
                namespace,
                item.key,
                {"category": category, "text": new_memory_text},
            )

        return {
            "answer": f"Updated your memory: {new_memory_text}.",
            "status": "updated",
        }

    # FORGET: Delete matching memories.
    forget_match = re.match(
        r"(?:please\s+)?(?:forget|delete|remove)\s+"
        r"(?:that\s+)?(.+)$",
        text,
        re.IGNORECASE,
    )

    if forget_match:
        subject = forget_match.group(1).strip()
        matches = _find_matching_memories(store, namespace, subject)

        if not matches:
            return {
                "answer": f"I couldn't find a stored memory about '{subject}'.",
                "status": "not_found",
            }

        for item in matches:
            store.delete(namespace, item.key)

        return {
            "answer": (
                f"Deleted {len(matches)} matching memory "
                f"record(s) about '{subject}'."
            ),
            "status": "deleted",
        }

    # REMEMBER: Save a new memory.
    remember_prefixes = (
        "please remember that ",
        "remember that ",
        "please remember ",
        "remember ",
        "save this memory: ",
        "keep in mind that ",
    )

    for prefix in remember_prefixes:
        if lower_text.startswith(prefix):
            memory_text = text[len(prefix):].strip()

            if not memory_text:
                return {
                    "answer": "Please specify what I should remember.",
                    "status": "invalid",
                }

            matches = _find_matching_memories(
                store, namespace, memory_text
            )

            if matches:
                return {
                    "answer": (
                        "A matching memory already exists. "
                        "Use an explicit update request to change it."
                    ),
                    "status": "already_exists",
                }

            memory_key = f"memory_{uuid4().hex}"
            store.put(
                namespace,
                memory_key,
                {
                    "category": "user_memory",
                    "text": memory_text,
                },
            )

            return {
                "answer": f"I've saved this memory: {memory_text}.",
                "status": "saved",
            }

    return None





def is_personal_memory_question(question: str) -> bool:
    """
    Identify personal-memory questions and explicit memory commands.

    Both route directly to finalize_node(). Explicit CRUD commands
    are processed before personal-memory questions there.
    """
    text = question.lower().strip()

    memory_commands = (
        "remember that ",
        "remember ",
        "please remember ",
        "please remember that ",
        "save this memory:",
        "keep in mind that ",
        "update ",
        "change ",
        "replace ",
        "forget ",
        "please forget ",
        "delete ",
        "remove ",
    )

    if text.startswith(memory_commands):
        return True

    # Natural-language preference corrections should be handled as
    # memory updates, not sent through ordinary research/RAG routing.
    if re.match(
        r"(?:i\s+)?(?:have\s+)?(?:switched|changed)\s+.+?\s+to\s+.+$",
        text,
    ):
        return True

    memory_phrases = (
        "my favorite",
        "my favourite",
        "my name",
        "my age",
        "my project",
        "my preference",
        "my preferences",
        "what do you remember about me",
        "what do you know about me",
        "what did i tell you",
        "what have i told you",
        "do you remember",
        "did i tell you",
        "what is my",
        "what's my",
        "who am i",
    )

    return any(phrase in text for phrase in memory_phrases)




def retrieve_memories_node(
    state: ResearchState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """Retrieve semantically relevant long-term memories for this user."""
    question = state.get("question", "").strip()
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    if not user_id:
        raise ValueError("user_id is required in config['configurable']")

    namespace = get_user_memory_namespace(user_id)
    logger.info("Long-term memory retrieval started | user_id=%s", user_id)
    try:
        items = store.search(namespace, query=question, limit=3)
        memories = []
        for item in items:
            value = item.value or {}
            text = value.get("text", "").strip()
            if text:
                category = value.get("category", "general")
                memories.append(f"[{category}] {text}")

        logger.info(
            "Long-term memory retrieval completed | matches=%d",
            len(memories),
        )

        return {
            "long_term_memories": memories,
            "memory_retrieval_status": "completed" if memories else "no_matches",
        }
    except Exception as exc:
        # Memory retrieval should not prevent the research workflow from running.
        logger.exception("Long-term memory retrieval failed")

        return {
            "long_term_memories": [],
            "memory_retrieval_status": f"failed: {type(exc).__name__}: {exc}",
        }




def _explicitly_requests_web_research(question: str) -> bool:
    """Return True when the question clearly asks for external/current research."""

    text = question.lower()

    web_intent_phrases = (
        "search the web",
        "search online",
        "search the internet",
        "look up online",
        "look it up online",
        "find online",
        "on the internet",
        "from the web",
        "web research",
        "online research",
        "external sources",
        "external references",
        "official documentation",
        "official docs",
        "official website",
        "latest",
        "current",
        "currently",
        "today",
        "right now",
        "this week",
        "recent news",
        "latest news",
        "verify online",
        "check online",
    )

    return any(phrase in text for phrase in web_intent_phrases)




def _filter_tool_calls(
    response,
    *,
    web_search_allowed: bool,
    web_search_already_attempted: bool,
):
    """
    Enforce the web-search policy after the model responds.

    The full registered tool list remains bound to the model, but a
    disallowed or repeated web_search request is removed before ToolNode
    can execute it. Also permits at most one web_search call in one
    assistant response.
    """

    tool_calls = getattr(response, "tool_calls", [])

    if not tool_calls:
        return response

    filtered_calls = []
    web_search_kept = False

    for call in tool_calls:
        tool_name = call.get("name")

        if tool_name == "web_search":
            if (
                not web_search_allowed
                or web_search_already_attempted
                or web_search_kept
            ):
                continue

            web_search_kept = True

        filtered_calls.append(call)

    if len(filtered_calls) != len(tool_calls):
        response = response.model_copy(
            update={"tool_calls": filtered_calls}
        )

    return response




# 2. Planner node
def planner_node(state: ResearchState) -> dict:
    """Create a research plan using the current question and conversation history."""

    question = state.get("question", "").strip()
    if not question:
        raise ValueError("Research question is missing from state.")

    messages = state.get("messages", [])
    previous_messages = messages[:-1] if messages else []

    history_lines = []
    for message in previous_messages[-8:]:
        role = getattr(message, "type", "unknown")
        content = getattr(message, "content", "")

        if role == "human":
            role = "User"
        elif role == "ai":
            role = "Assistant"
        else:
            continue

        if isinstance(content, str) and content.strip():
            history_lines.append(f"{role}: {content}")

    conversation_history = (
        "\n".join(history_lines)
        if history_lines
        else "No previous conversation."
    )

    long_term_memories = state.get("long_term_memories", [])
    memory_context = (
        "\n".join(f"- {memory}" for memory in long_term_memories)
        if long_term_memories
        else "No relevant long-term memories were retrieved."
    )
    logger.info("Planner execution started")
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an intelligent research planner operating "
                    "inside a LangGraph research agent.\n\n"
                    "Your responsibility is to understand the user's "
                    "latest question and decide what information is needed.\n\n"
                    "CONTEXT PRIORITY RULES (highest to lowest):\n"
                    "1. Follow the latest user message and its explicit corrections. "
                    "The current user message overrides older context and saved memories.\n"
                    "2. Use recent conversation history to resolve follow-ups, "
                    "pronouns, omitted details, and references such as 'that one' "
                    "or 'what about 20 days?'.\n"
                    "3. If the latest message does not override the conversation, "
                    "use the most recent explicit user statements in this thread.\n"
                    "4. Use retrieved long-term memories as durable user context "
                    "when they are relevant and do not conflict with newer user statements.\n"
                    "5. Do not treat an assistant's earlier assumption as a confirmed "
                    "user fact. Never invent personal facts.\n"
                    "6. If a personal fact is absent from both relevant conversation "
                    "history and long-term memory, communicate that it is unavailable.\n"
                    "7. Do not suggest searching emails, Jira, external systems, "
                    "or documents unless the user explicitly requests it.\n\n"
                    "CONVERSATION RULES:\n"
                    "1. Use the previous conversation to understand incomplete follow-ups.\n"
                    "2. Prefer the most recent explicit user correction when user "
                    "statements conflict.\n"
                    "3. Do not ask for clarification when the intended context is clear.\n\n"
                    "RESEARCH RULES:\n"
                    "1. Use only the research steps that are actually necessary.\n"
                    "2. Do not suggest tools for questions that can be answered "
                    "from available memory or general knowledge.\n"
                    "3. Do not suggest document searches for unrelated questions.\n"
                    "4. Do not suggest web searches unless external research is needed.\n\n"
                    "Return a numbered list of 1 to 3 concise actionable steps.\n"
                    "Do not greet the user.\n"
                    "Do not answer the question."
                )
            ),
            HumanMessage(
                content=(
                    f"Previous conversation:\n{conversation_history}\n\n"
                    f"Retrieved long-term memories:\n{memory_context}\n\n"
                    f"Latest user question:\n{question}"
                )
            ),
        ]
    )

    raw_plan = str(response.content).strip()
    plan = []

    for line in raw_plan.splitlines():
        cleaned = line.strip()
        if cleaned:
            cleaned = cleaned.lstrip("0123456789").lstrip(". )-")
            if cleaned:
                plan.append(cleaned)

    if not plan:
        plan = [
            f"Understand the user's question: {question}",
            "Determine whether local documents, web research, "
            "or calculation is needed.",
            "Prepare a clear and accurate answer.",
        ]
    logger.info("Planner execution completed | plan_steps=%d", len(plan))
    return {
        "plan": plan,
        "current_task": plan[0],
        "status": "planning_complete",
    }




# 3. Agent node
def agent_node(state: ResearchState) -> dict:
    """Decide whether to answer directly or call a research tool."""

    question = state.get("question", "").strip()

    if not question:
        raise ValueError(
            "Research question is missing from state."
        )

    plan = state.get("plan", [])
    long_term_memories = state.get("long_term_memories", [])
    memory_context = (
        "\n".join(f"- {memory}" for memory in long_term_memories)
        if long_term_memories
        else "No relevant long-term memories were retrieved."
    )

    system_prompt = SystemMessage(
        content=(
            "You are a research assistant operating inside "
            "a LangGraph application.\n\n"
            "Answer the user's actual question directly. "
            "Use available evidence and never invent tool results.\n\n"
            f"User question:\n{question}\n\n"
            "Research plan:\n"
            + "\n".join(f"- {task}" for task in plan)
            + "\n\n"
            + f"Relevant long-term user memories:\n{memory_context}\n\n"
            + "CONTEXT PRIORITY (highest to lowest):\n"
            + "1. Follow the latest user message, including explicit corrections.\n"
            + "2. Use recent conversation history to resolve follow-ups and references. "
            + "If the user explicitly corrected or changed something earlier in this "
            + "thread, use the most recent explicit user statement.\n"
            + "3. Use long-term memories as durable user context only when they do not "
            + "conflict with newer statements in the conversation.\n"
            + "4. Do not treat an assistant's previous assumption as a confirmed user fact.\n"
            + "5. Do not invent missing personal details.\n\n"
            + "Long-term memories are user-specific context, not verified external facts. "
            + "A current user correction must never be overridden by an older memory.\n\n"
            + "Available tools:\n"
            "- web_search(query, max_results): search the web "
            "for external information.\n"
            "- calculator(expression): perform arithmetic.\n"
            "- document_search(query, max_results): search the "
            "indexed local PDF and TXT documents.\n\n"
            "Tool-selection rules (follow these in priority order):\n"
            "1. Answer general knowledge and conceptual questions "
            "directly from your existing knowledge. Examples include "
            "definitions, simple explanations, and comparisons of "
            "common technical concepts.\n"
            "2. Do NOT use web_search for basic definitions, educational "
            "explanations, ordinary programming concepts, or general "
            "conceptual comparisons unless the user explicitly asks "
            "for online research or current information.\n"
            "3. Use document_search when the user asks about their local "
            "documents, notes, uploaded files, or project-specific "
            "indexed knowledge. Do not use it for unrelated general "
            "knowledge questions.\n"
            "4. Use calculator when the question requires arithmetic.\n"
            "5. Use web_search only when the user explicitly asks to "
            "search the web/internet, find information online, consult "
            "external sources, verify something online, or asks for "
            "latest/current/today/recent information or official "
            "documentation/website information.\n"
            "6. If the question can be answered accurately without a "
            "tool, answer directly and do not call a tool.\n"
            "7. Use multiple registered tools only when each is needed.\n"
            "8. After receiving a tool result, inspect it before responding.\n"
            "9. For a single research question, call web_search at most once.\n"
            "10. Use a descriptive search query, never a result URL.\n"
            "11. Set web_search max_results to 5 or fewer.\n"
            "12. Never call web_open, browser, open_url, or any tool "
            "that is not registered.\n"
            "13. After web_search returns, use its results directly. "
            "Do not attempt to open or follow individual links.\n"
            "14. If web_search fails, disclose the failure rather than "
            "repeating the same search.\n"
            "15. Only call tools explicitly available in the current request.\n"
        )
    )

    messages = state.get("messages", [])

    logger.info("Agent execution started")

    # Invoke the model with the complete registered tool list.
    response = model_with_tools.invoke(
        [system_prompt, *messages]
    )

    # A web search is allowed only for an explicit external/current
    # research request. The model remains bound to all tools so Groq
    # does not receive a changing tool schema.
    web_search_allowed = _explicitly_requests_web_research(question)

    # Tool results are accumulated in ResearchState. Check whether a
    # web_search response has already been recorded for this question.
    tool_results = state.get("tool_results", [])
    web_search_already_attempted = any(
        str(result).startswith("web_search:")
        for result in tool_results
    )

    response = _filter_tool_calls(
        response,
        web_search_allowed=web_search_allowed,
        web_search_already_attempted=web_search_already_attempted,
    )

    logger.info(
        "Agent decision completed | tool_calls=%d",
        len(response.tool_calls),
    )


    return {
        "messages": [response],
        "iteration_count": (
            state.get("iteration_count", 0) + 1
        ),
        "status": (
            "tool_requested"
            if response.tool_calls
            else "completed"
        ),
    }




# Human-in-the-Loop approval node
# Human-in-the-Loop approval node
def human_review_node(state: ResearchState) -> dict:
    """
    Pause execution and ask the human to approve pending tool calls.

    This node must remain free of external side effects because
    LangGraph may restart it when an interrupted graph resumes.
    """

    logger.info("NODE: HUMAN_REVIEW - Started")

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No messages found for human review.")

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        raise ValueError("No pending tool calls found for approval.")

    approval_request = {
        "type": "human_approval_required",
        "message": "The agent wants to execute the following tool calls.",
        "tool_calls": [
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "arguments": call.get("args", {}),
            }
            for call in tool_calls
        ],
    }

    logger.info(
        "Human approval requested | tool_calls=%d | names=%s",
        len(tool_calls),
        [call.get("name") for call in tool_calls],
    )

    # Pause execution until Command(resume=...) is supplied.
    decision = interrupt(approval_request)

    if not isinstance(decision, dict):
        raise ValueError("Approval decision must be a dictionary.")

    approved = decision.get("approved")

    if not isinstance(approved, bool):
        raise ValueError(
            "Approval decision must contain a boolean 'approved'."
        )

    logger.info(
        "HUMAN DECISION RECEIVED | approved=%s",
        approved,
    )

    return {
        "human_approved": approved,
        "approval_status": "approved" if approved else "rejected",
        "status": "approval_granted" if approved else "approval_rejected",
    }


# Handle rejected tool calls
def reject_tool_calls_node(state: ResearchState) -> dict:
    """
    Stop the workflow when the human rejects tool execution.
    No rejected tool is executed and the agent is not called again.
    """

    logger.info("NODE: REJECT_TOOL_CALLS - Started")

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No messages found for rejection.")

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        raise ValueError("No pending tool calls found.")

    rejection_messages = []

    for call in tool_calls:
        rejection_messages.append(
            ToolMessage(
                content=(
                    "Tool execution was rejected by the human reviewer. "
                    "The requested tool was not executed."
                ),
                tool_call_id=call["id"],
                name=call.get("name"),
            )
        )

    logger.warning(
        "TOOL EXECUTION REJECTED | tool_count=%d | tools=%s",
        len(tool_calls),
        [call.get("name") for call in tool_calls],
    )

    return {
        "messages": rejection_messages,
        "status": "tool_rejected",
        "approval_status": "rejected",
        "human_approved": False,
        "final_answer": (
            "Tool execution was rejected. "
            "The requested operation was not performed."
        ),
    }

# 4. Tools node
def tools_node(state: ResearchState, config: RunnableConfig | None = None) -> dict:
    """Execute approved tool calls and record results or errors."""
    logger.info("NODE: TOOLS - Started")

    output = tool_executor.invoke(state, config=config)
    tool_messages = output.get("messages", [])
    tool_results = []
    errors = []
    has_error = False

    for message in tool_messages:
        name = getattr(message, "name", "tool")
        content = str(getattr(message, "content", ""))
        tool_results.append(f"{name}: {content}")

        error_markers = (
            "WEB_SEARCH_FAILED",
            "Calculation error:",
            "RAG_SEARCH_FAILED",
            "NO_RELEVANT_DOCUMENTS",
            "No web search results were found",
        )
        if any(marker in content for marker in error_markers):
            errors.append(content)
            has_error = True

    status = "tool_error" if has_error else "tool_completed"
    logger.info(
        "Tools execution completed | status=%s | results=%d | errors=%d",
        status,
        len(tool_results),
        len(errors),
    )

    return {
        "messages": tool_messages,
        "tool_results": tool_results,
        "errors": errors,
        "status": status,
    }



def _extract_valid_final_answer(content, max_length: int = 20000) -> str:
    """Extract a non-empty, bounded text answer from an LLM response."""
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    text_parts.append(block.strip())
            elif isinstance(block, dict):
                block_type = block.get("type")
                block_text = block.get("text")
                if block_type in (None, "text") and isinstance(block_text, str):
                    if block_text.strip():
                        text_parts.append(block_text.strip())
        answer = "\\n".join(text_parts).strip()
    else:
        answer = ""

    if not answer:
        return (
            "I couldn't generate a complete answer for this request. "
            "Please try again."
        )

    if len(answer) > max_length:
        return (
            "The generated answer exceeded the allowed output length, "
            "so I couldn't return it safely. Please narrow the request "
            "and try again."
        )

    return answer


# 5. Finalization node

def finalize_node(
    state: ResearchState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """Generate a final answer using recorded research evidence."""

    # Rejection safeguard: do not generate answer if tool approval was rejected
    # Reject only when the human explicitly rejected tool execution.
    if state.get("approval_status") == "rejected":
        cancellation_text = (
            state.get("final_answer")
            or "Tool execution was rejected by the user. The request has been cancelled."
        )

        return {
            "final_answer": cancellation_text,
            "messages": [AIMessage(content=cancellation_text)],
            "status": "rejected",
            "approval_status": "rejected",
            "memory_write_status": "not_requested",
        }

    question = state.get("question", "").strip()
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")

    if not user_id:
        raise ValueError("user_id is required for memory operations")

    # 1. Explicit memory CRUD operations must always run first.
    memory_result = _handle_memory_request(
        question=question,
        store=store,
        user_id=user_id,
    )

    if memory_result is not None:
        answer = memory_result["answer"]

        logger.info(
            "Memory operation completed | status=%s",
            memory_result["status"],
        )

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "status": "completed",
            "memory_write_status": memory_result["status"],
        }

    # 2. Personal-memory questions must use retrieved memories only.
    
    # 2. Personal-memory questions: check current conversation first,
    # then fall back to long-term memories.
    if is_personal_memory_question(question):
        memories = state.get("long_term_memories", [])
        messages = state.get("messages", [])
        question_lower = question.lower()

        ignored_words = {
            "what", "is", "my", "the", "a", "an", "do", "you",
            "know", "remember", "about", "me", "did", "i", "tell",
            "have", "please", "can", "could", "would", "was", "are",
            "who", "your", "to", "for", "of",
        }

        question_words = {
            word
            for word in re.findall(r"[a-z0-9]+", question_lower)
            if word not in ignored_words
        }

        # First: search previous user messages in this thread.
        conversation_matches = []

        for message in messages[:-1]:
            if not isinstance(message, HumanMessage):
                continue

            previous_text = str(message.content).strip()
            previous_words = set(
                re.findall(r"[a-z0-9]+", previous_text.lower())
            )

            if question_words and question_words.issubset(previous_words):
                conversation_matches.append(previous_text)

        if conversation_matches:
            answer = (
                "Based on our conversation, you said: "
                + conversation_matches[-1]
            )

        else:
            # Second: fall back to long-term memories.
            if (
                "what do you remember about me" in question_lower
                or "what do you know about me" in question_lower
            ):
                if memories:
                    answer = (
                        "Here is the information I found in your saved memories:\n"
                        + "\n".join(f"- {memory}" for memory in memories)
                    )
                else:
                    answer = (
                        "I don't have relevant information in your saved "
                        "memories yet."
                    )

            else:
                matching_memories = []

                for memory in memories:
                    memory_words = set(
                        re.findall(r"[a-z0-9]+", memory.lower())
                    )

                    if question_words and question_words.issubset(memory_words):
                        matching_memories.append(memory)

                if matching_memories:
                    answer = (
                        "Based on your saved memory:\n"
                        + "\n".join(f"- {memory}" for memory in matching_memories)
                    )
                else:
                    answer = (
                        "I don't have this information in my conversation "
                        "history or saved memories, so I don't want to guess."
                    )

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "status": "completed",
            "memory_write_status": "not_requested",
        }



    # 3. Ordinary research finalization.
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])
    errors = state.get("errors", [])

    research_context = (
        "\n".join(f"- {result}" for result in tool_results)
        if tool_results
        else "No tool results were recorded."
    )

    error_context = (
        "\n".join(f"- {error}" for error in errors)
        if errors
        else "No tool errors were recorded."
    )

    long_term_memories = state.get("long_term_memories", [])
    memory_context = (
        "\n".join(f"- {memory}" for memory in long_term_memories)
        if long_term_memories
        else "No relevant long-term memories were retrieved."
    )

    messages = state.get("messages", [])
    last_user_index = max(
        (
            i for i, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ),
        default=len(messages),
    )

    previous_messages = messages[:last_user_index]
    conversation_history = "\n".join(
        f"{message.type}: {str(message.content)}"
        for message in previous_messages[-12:]
    )

    logger.info("Final answer generation started")
    
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research assistant preparing a final "
                    "answer. Answer the original question directly.\n\n"
                    "Use the recorded tool results as evidence. "
                    "Apply this context priority: (1) the latest user message and "
                    "its explicit corrections, (2) the most recent explicit user "
                    "statements in the conversation history, (3) relevant long-term "
                    "memories that do not conflict with newer statements. "
                    "Use conversation history to resolve follow-up references. "
                    "Treat long-term memories as user-specific context, not verified "
                    "external facts. Do not treat an assistant's earlier assumption "
                    "as a confirmed user fact, and never invent personal details. "
                    "When document_search returns relevant passages, "
                    "base claims about those documents on the passages "
                    "actually returned. Mention source filenames when "
                    "they are available and useful.\n\n"
                    "Do not claim a tool succeeded if its output "
                    "indicates failure. Do not invent facts, URLs, "
                    "citations, or document contents. If the available "
                    "evidence is insufficient, state that limitation "
                    "clearly. You do not have tools available in this "
                    "finalization step.\n\n"
                    "Classify evidence by its tool name:\n"
                    "- document_search results are from local documents.\n"
                    "- web_search results are from external web searches.\n"
                    "- calculator results are computed values.\n\n"
                    "Never attribute web-search information to local "
                    "documents. Never claim that a URL was opened or its "
                    "full contents were verified when only a search snippet "
                    "was returned. When external search results are "
                    "available, identify the relevant result title and URL. "
                    "If web search failed, explicitly disclose that live "
                    "verification was unavailable."
                )
            ),
            HumanMessage(
                content=(
                    f"Original question:\n{question}\n\n"
                    f"Research plan:\n{plan}\n\n"
                    f"Recorded tool results:\n{research_context}\n\n"
                    f"Recorded tool errors:\n{error_context}\n\n"
                    f"Relevant long-term user memories:\n{memory_context}\n\n"
                    f"Previous conversation:\n{conversation_history}\n\n"
                    "Prepare the final answer from this information. "
                    "If no tools were needed, answer directly from "
                    "your general knowledge."
                )
            ),
        ]
    )

    answer_text = _extract_valid_final_answer(response.content)
    logger.info(
        "Final answer generated | answer_length=%d",
        len(answer_text),
    )
    return {
        "final_answer": answer_text,
        "messages": [AIMessage(content=answer_text)],
        "status": "completed",
        "memory_write_status": "not_requested",
    }
