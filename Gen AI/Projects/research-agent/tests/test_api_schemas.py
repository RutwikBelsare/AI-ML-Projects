
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ChatRequest,
    ApprovalRequest,
    ChatResponse,
)


def test_valid_chat_request():
    request = ChatRequest(
        question="What is LangGraph?",
        thread_id="thread_123",
        user_id="local_user",
    )

    assert request.question == "What is LangGraph?"
    assert request.thread_id == "thread_123"


def test_empty_question_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="")


def test_question_over_5000_characters_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="A" * 5001)


def test_valid_approval_request():
    request = ApprovalRequest(
        thread_id="thread_123",
        approved=True,
    )

    assert request.approved is True


def test_chat_response():
    response = ChatResponse(
        thread_id="thread_123",
        user_id="local_user",
        status="completed",
        final_answer="LangGraph is a framework.",
    )

    assert response.status == "completed"
    assert response.approval_required is False
