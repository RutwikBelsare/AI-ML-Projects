import pytest
from unittest.mock import AsyncMock

from backend.core.exceptions import LLMException
from backend.services.support_services import SupportService


# ---------------------------------------------------------
# Mock LLM Service
# ---------------------------------------------------------

@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    return mock


# ---------------------------------------------------------
# Valid Structured Response
# ---------------------------------------------------------

VALID_RESPONSE = """
{
    "category": "order",
    "intent": "order_status",
    "sentiment": "negative",
    "priority": "high",
    "response": "We are sorry about the delay. Please check your tracking information.",
    "requires_human": false
}
"""


# ---------------------------------------------------------
# Test 1: Valid Structured Response
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_structured_response(mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE

    service = SupportService(llm_service=mock_llm)

    result = await service.process_message(
        "Where is my order? It has not arrived yet."
    )

    assert result.category == "order"
    assert result.intent == "order_status"
    assert result.sentiment == "negative"
    assert result.priority == "high"
    assert result.requires_human is False


# ---------------------------------------------------------
# Test 2: Invalid JSON Response
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_json_response(mock_llm):
    mock_llm.generate.return_value = """
    This is not valid JSON.
    """

    service = SupportService(llm_service=mock_llm)

    with pytest.raises(LLMException) as exc_info:
        await service.process_message(
            "Where is my order?"
        )

    assert (
        exc_info.value.message
        == "The AI service returned an invalid response format."
    )


# ---------------------------------------------------------
# Test 3: Invalid Category
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_category(mock_llm):
    mock_llm.generate.return_value = """
    {
        "category": "random_category",
        "intent": "order_status",
        "sentiment": "neutral",
        "priority": "medium",
        "response": "Your order is being processed.",
        "requires_human": false
    }
    """

    service = SupportService(llm_service=mock_llm)

    with pytest.raises(LLMException):
        await service.process_message(
            "Where is my order?"
        )


# ---------------------------------------------------------
# Test 4: Missing Required Field
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_required_field(mock_llm):
    mock_llm.generate.return_value = """
    {
        "category": "order",
        "intent": "order_status",
        "sentiment": "neutral",
        "priority": "medium",
        "response": "Your order is being processed."
    }
    """

    service = SupportService(llm_service=mock_llm)

    with pytest.raises(LLMException):
        await service.process_message(
            "Where is my order?"
        )


# ---------------------------------------------------------
# Test 5: Valid Intent
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_intent(mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE

    service = SupportService(llm_service=mock_llm)

    result = await service.process_message(
        "Where is my order?"
    )

    assert result.intent == "order_status"


# ---------------------------------------------------------
# Test 6: Invalid Intent
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_intent(mock_llm):
    mock_llm.generate.return_value = """
    {
        "category": "order",
        "intent": "random_intent",
        "sentiment": "neutral",
        "priority": "medium",
        "response": "Your order is being processed.",
        "requires_human": false
    }
    """

    service = SupportService(llm_service=mock_llm)

    with pytest.raises(LLMException):
        await service.process_message(
            "Where is my order?"
        ) 


# ---------------------------------------------------------
# Test 7: Support Service Streaming
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_message(mock_llm):

    # Classification response required because
    # stream_message() now classifies the message first.
    mock_llm.generate.return_value = """
    {
        "category": "order",
        "intent": "order_status",
        "sentiment": "negative",
        "priority": "medium",
        "requires_human": false
    }
    """

    async def mock_stream(message):

        yield "Your"
        yield " order"
        yield " is"
        yield " being"
        yield " processed."

    mock_llm.stream = mock_stream

    service = SupportService(
        llm_service=mock_llm
    )

    chunks = []

    async for chunk in service.stream_message(
        "Where is my order?"
    ):
        chunks.append(chunk)

    assert chunks == [
        "Your",
        " order",
        " is",
        " being",
        " processed.",
    ]

# ---------------------------------------------------------
# Test 8: Valid Classification
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_classification(mock_llm):

    mock_llm.generate.return_value = """
    {
        "category": "refund",
        "intent": "refund_request",
        "sentiment": "neutral",
        "priority": "medium",
        "requires_human": false
    }
    """

    service = SupportService(
        llm_service=mock_llm
    )

    result = await service.classify_message(
        "I want a refund."
    )

    assert result.category == "refund"
    assert result.intent == "refund_request"
    assert result.sentiment == "neutral"
    assert result.priority == "medium"
    assert result.requires_human is False


# ---------------------------------------------------------
# Test 9: Invalid Classification Intent
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_classification_intent(
    mock_llm,
):

    mock_llm.generate.return_value = """
    {
        "category": "refund",
        "intent": "order_tracking",
        "sentiment": "neutral",
        "priority": "medium",
        "requires_human": false
    }
    """

    service = SupportService(
        llm_service=mock_llm
    )

    with pytest.raises(LLMException):

        await service.classify_message(
            "I want a refund."
        )


# ---------------------------------------------------------
# Test 10: Intent-Aware Streaming
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_message_uses_classification(
    mock_llm,
):

    mock_llm.generate.return_value = """
    {
        "category": "refund",
        "intent": "refund_request",
        "sentiment": "neutral",
        "priority": "medium",
        "requires_human": false
    }
    """

    async def mock_stream(message):

        assert "refund_request" in message
        assert "refund" in message.lower()

        yield "I can help "
        yield "you with "
        yield "your refund request."

    mock_llm.stream = mock_stream

    service = SupportService(
        llm_service=mock_llm
    )

    chunks = []

    async for chunk in service.stream_message(
        "I want a refund."
    ):

        chunks.append(chunk)

    assert chunks == [
        "I can help ",
        "you with ",
        "your refund request.",
    ]