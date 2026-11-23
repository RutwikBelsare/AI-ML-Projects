import pytest
from unittest.mock import AsyncMock, Mock

from backend.services.tool_calling_service import (
    ToolCallingService,
)


@pytest.mark.asyncio
async def test_streaming_tool_calling():

    mock_llm = Mock()

    # =================================================
    # First call:
    # LLM decides that a tool is required.
    # =================================================

    mock_llm.chat_with_tools = AsyncMock(
        return_value={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_order_status",
                            "arguments": {
                                "order_id": "ORD1001"
                            },
                        }
                    }
                ],
            }
        }
    )

    # =================================================
    # Second call:
    # LLM streams the final response.
    # =================================================

    async def final_stream(*args, **kwargs):

        yield {
            "message": {
                "role": "assistant",
                "content": "Your order ",
            }
        }

        yield {
            "message": {
                "role": "assistant",
                "content": "ORD1001 is shipped.",
            }
        }

    # IMPORTANT:
    # stream_chat_with_tools() returns an async generator,
    # so it should NOT be an AsyncMock.
    mock_llm.stream_chat_with_tools = (
        lambda *args, **kwargs: final_stream()
    )

    service = ToolCallingService(
        llm_service=mock_llm
    )

    chunks = []

    async for chunk in service.stream_message(
        "Where is my order ORD1001?"
    ):
        chunks.append(chunk)

    assert chunks == [
        "Your order ",
        "ORD1001 is shipped.",
    ]

    assert (
        mock_llm.chat_with_tools.call_count
        == 1
    )