import pytest
from unittest.mock import AsyncMock, Mock

from backend.services.tool_calling_service import ToolCallingService


@pytest.fixture
def mock_llm():
    return Mock()


@pytest.mark.asyncio
async def test_tool_calling_executes_order_tool(mock_llm):

    mock_llm.chat_with_tools = AsyncMock(
        side_effect=[
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "get_order_status",
                                "arguments": {
                                    "order_id": "ORD1001"
                                },
                            },
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "Your order ORD1001 is currently "
                        "shipped with DHL."
                    ),
                }
            },
        ]
    )

    service = ToolCallingService(
        llm_service=mock_llm
    )

    result = await service.process_message(
        "Where is my order ORD1001?"
    )

    assert "ORD1001" in result
    assert "shipped" in result

    assert mock_llm.chat_with_tools.call_count == 2


@pytest.mark.asyncio
async def test_no_tool_call_returns_llm_response(mock_llm):

    mock_llm.chat_with_tools = AsyncMock(
        return_value={
            "message": {
                "role": "assistant",
                "content": "How can I help you?",
                "tool_calls": [],
            }
        }
    )

    service = ToolCallingService(
        llm_service=mock_llm
    )

    result = await service.process_message(
        "Hello"
    )

    assert result == "How can I help you?"

    assert mock_llm.chat_with_tools.call_count == 1


@pytest.mark.asyncio
async def test_invalid_tool_arguments(mock_llm):

    mock_llm.chat_with_tools = AsyncMock(
        return_value={
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_order_status",
                            "arguments": "{invalid-json}",
                        },
                    }
                ],
            }
        }
    )

    service = ToolCallingService(
        llm_service=mock_llm
    )

    with pytest.raises(Exception):
        await service.process_message(
            "Where is my order ORD1001?"
        )


@pytest.mark.asyncio
async def test_unknown_tool(mock_llm):

    mock_llm.chat_with_tools = AsyncMock(
        return_value={
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "delete_database",
                            "arguments": {},
                        },
                    }
                ],
            }
        }
    )

    service = ToolCallingService(
        llm_service=mock_llm
    )

    with pytest.raises(Exception):
        await service.process_message(
            "Delete the database."
        )


@pytest.mark.asyncio
async def test_empty_final_response(mock_llm):

    mock_llm.chat_with_tools = AsyncMock(
        side_effect=[
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "get_order_status",
                                "arguments": {
                                    "order_id": "ORD1001"
                                },
                            },
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                }
            },
        ]
    )

    service = ToolCallingService(
        llm_service=mock_llm
    )

    with pytest.raises(Exception):
        await service.process_message(
            "Where is my order ORD1001?"
        )