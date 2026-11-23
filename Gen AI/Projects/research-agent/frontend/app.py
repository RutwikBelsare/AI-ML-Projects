import html
import streamlit as st
import requests
import uuid
import json


# CONFIGURATION


API_BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
)





# CUSTOM CHAT UI STYLING
st.markdown(
    """
    <style>

    /* User message row */
    .user-message-row {
        display: flex;
        
        justify-content: flex-end;
        width: 100%;
        margin: 12px 0 20px 0;
    }

    /* User message bubble */
    .user-message-bubble {
        max-width: 80%;
        padding: 12px 18px;
        border-radius: 18px;
        color: var(--text-color);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-size: 16px;
        line-height: 1;
    }

    /* Assistant response */
    .assistant-message {
        width: 100%;
        text-align: left;
        margin: 12px 0 24px 0;
    }

    /* Three bouncing dots */
    .typing-loader {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 4px;
        height: 30px;
    }

    .typing-loader span {
        width: 8px;
        height: 8px;
        background-color: #888888;
        border-radius: 50%;
        animation: bounce 1.2s infinite ease-in-out;
    }

    .typing-loader span:nth-child(1) {
        animation-delay: 0s;
    }

    .typing-loader span:nth-child(2) {
        animation-delay: 0.2s;
    }

    .typing-loader span:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes bounce {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.5;
        }

        30% {
            transform: translateY(-8px);
            opacity: 1;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# SESSION STATE
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = False

if "user_id" not in st.session_state:
    st.session_state.user_id = "local_user"


# BACKEND HELPER FUNCTIONS
def fetch_conversations():
    """Fetch all saved conversations for the current user."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/conversations",
            params={"user_id": st.session_state.user_id},
            timeout=10,
        )

        response.raise_for_status()

        return response.json().get("conversations", [])

    except requests.RequestException as exc:
        st.warning(f"Unable to load conversation history: {exc}")
        return []


def load_conversation(thread_id):
    """Load a selected conversation and its messages."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/conversations/{thread_id}",
            params={"user_id": st.session_state.user_id},
            timeout=10,
        )

        response.raise_for_status()

        conversation = response.json()

        st.session_state.thread_id = conversation["thread_id"]

        st.session_state.messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in conversation.get("messages", [])
        ]

        st.session_state.pending_approval = False

        return True

    except requests.RequestException as exc:
        st.error(f"Unable to open conversation: {exc}")
        return False



# SIDEBAR
with st.sidebar:

    st.title("🔬 Research Agent")
    st.caption("LangGraph + RAG + Long-term Memory")

    # New conversation button
    if st.button(
        "＋ New Conversation",
        use_container_width=True,
        disabled=st.session_state.pending_approval,
    ):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.session_state.pending_approval = False
        st.rerun()

    st.divider()

    # Recent conversations
    st.subheader("Recent Chats")

    conversations = fetch_conversations()

    if conversations:

        for conversation in conversations:

            thread_id = conversation["thread_id"]
            title = conversation.get("title", "New Chat")

            # Highlight the currently selected conversation
            is_selected = (
                st.session_state.thread_id == thread_id
            )

            button_label = (
                f"● {title}"
                if is_selected
                else title
            )

            if st.button(
                button_label,
                key=f"conversation_{thread_id}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                disabled=st.session_state.pending_approval,
            ):

                if load_conversation(thread_id):
                    st.rerun()

    else:
        st.caption("No previous conversations yet.")

    if st.session_state.pending_approval:
        st.caption(
            "Finish the pending approval before switching chats."
        )

    st.divider()


# MAIN CHAT INTERFACE

st.caption(
    "Ask questions, research topics, and analyze information."
)



# Display saved conversation messages
for message in st.session_state.messages:

    if message["role"] == "user":

        escaped_content = html.escape(message["content"])

        st.markdown(
            f"""
            <div class="user-message-row">
                <div class="user-message-bubble">
                    {escaped_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f'<div class="assistant-message"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(message["content"])


# APPROVAL HANDLING
if st.session_state.pending_approval:

    st.warning(
        "Human approval is required before tool execution."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Approve",
            type="primary",
            use_container_width=True,
        ):

            try:

                response = requests.post(
                    f"{API_BASE_URL}/chat/approve",
                    json={
                        "thread_id": st.session_state.thread_id,
                        "user_id": st.session_state.user_id,
                        "approved": True,
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

                st.session_state.pending_approval = False

                answer = result.get("final_answer", "")

                if answer:

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                st.rerun()

            except requests.RequestException as exc:

                st.error(f"Approval failed: {exc}")

    with col2:

        if st.button(
            "Reject",
            use_container_width=True,
        ):

            try:

                response = requests.post(
                    f"{API_BASE_URL}/chat/approve",
                    json={
                        "thread_id": st.session_state.thread_id,
                        "user_id": st.session_state.user_id,
                        "approved": False,
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

                st.session_state.pending_approval = False

                answer = result.get("final_answer", "")

                if answer:

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                st.rerun()

            except requests.RequestException as exc:

                st.error(f"Rejection failed: {exc}")



# CHAT INPUT
if not st.session_state.pending_approval:

    chat_submission = st.chat_input(
        "Ask your research question...",
        accept_file="multiple",
        file_type=["pdf", "txt", "docx", "csv"],
        max_upload_size=20,
    )

    question = ""

    if chat_submission:

        question = (chat_submission.text or "").strip()

        uploaded_files = chat_submission.files

        upload_failed = False

        # Upload attached documents before running the agent.
        for uploaded_file in uploaded_files:

            with st.spinner(
                f"Uploading and indexing {uploaded_file.name}..."
            ):

                try:

                    response = requests.post(
                        f"{API_BASE_URL}/documents/upload",

                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                                or "application/octet-stream",
                            )
                        },

                        timeout=(10, 180),
                    )

                    response.raise_for_status()

                    result = response.json()

                    document = result.get("document", {})

                    if result.get("status") == "duplicate":

                        st.info(
                            f"{uploaded_file.name} is already indexed."
                        )

                    else:

                        st.success(
                            f"{uploaded_file.name} uploaded and indexed successfully."
                        )

                    st.caption(
                        f"Chunks indexed: "
                        f"{document.get('chunks_indexed', 0)}"
                    )

                except requests.RequestException as exc:

                    upload_failed = True

                    error_detail = ""

                    if exc.response is not None:
                        try:
                            error_detail = exc.response.json().get(
                                "detail", ""
                            )
                        except ValueError:
                            error_detail = exc.response.text

                    st.error(
                        f"Upload failed for {uploaded_file.name}: "
                        f"{error_detail or str(exc)}"
                    )

        # Do not execute the question if an attached document failed.
        if upload_failed:
            question = ""

    if question:


        # Generate a new thread ID if this is a new conversation
        if st.session_state.thread_id is None:

            st.session_state.thread_id = (
                f"thread_{uuid.uuid4().hex[:12]}"
            )

        # Save user message in Streamlit session
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        escaped_question = html.escape(question)

        st.markdown(
            f"""
            <div class="user-message-row">
                <div class="user-message-bubble">
                    {escaped_question}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():

            status_placeholder = st.empty()
            answer_placeholder = st.empty()

            accumulated_answer = ""

            # Display animated three-dot loader
            status_placeholder.markdown(
                """
                <div class="typing-loader">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:

                response = requests.post(
                    f"{API_BASE_URL}/chat/stream",
                    json={
                        "question": question,
                        "thread_id": st.session_state.thread_id,
                        "user_id": st.session_state.user_id,
                    },
                    stream=True,
                    timeout=(10, 300),
                )

                response.raise_for_status()

                current_event = None
                completed_data = None
                approval_data = None
                error_message = None

                # Process Server-Sent Events
                for line in response.iter_lines(
                    decode_unicode=True
                ):

                    if not line:
                        continue

                    if line.startswith("event:"):

                        current_event = line[6:].strip()

                    elif line.startswith("data:"):

                        data = json.loads(
                            line[5:].strip()
                        )

                        if current_event == "node_update":

                            # Keep the animated loader visible.
                            # Do not expose internal backend node names.
                            pass

                        elif current_event == "token":

                            # Remove loader when the first response token arrives
                            status_placeholder.empty()

                            accumulated_answer += data.get(
                                "content", ""
                            )

                            answer_placeholder.markdown(
                                accumulated_answer
                            )

                        elif current_event == "approval_required":

                            approval_data = data

                        elif current_event == "completed":

                            completed_data = data

                        elif current_event == "error":

                            error_message = data.get(
                                "message",
                                "Research execution failed.",
                            )

                # Handle final execution state

                if error_message:

                    status_placeholder.empty()
                    status_placeholder.error(error_message)

                elif approval_data:

                    status_placeholder.empty()

                    st.session_state.pending_approval = True

                    st.warning(
                        "Human approval is required before tool execution."
                    )

                    st.info(
                        approval_data.get("content", "")
                    )

                elif completed_data:

                    status_placeholder.empty()

                    final_answer = completed_data.get(
                        "final_answer",
                        accumulated_answer,
                    )

                    if final_answer:

                        answer_placeholder.markdown(
                            final_answer
                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_answer,
                        })

                # Refresh sidebar and conversation history
                st.rerun()

            except requests.RequestException as exc:

                status_placeholder.error(
                    f"Backend connection failed: {exc}"
                )

            except Exception as exc:

                status_placeholder.error(
                    f"Unexpected error: {exc}"
                )
