import pytest
import httpx

from backend.services.ollama_service import OllamaService
from backend.core.exceptions import (
    LLMException,
    LLMTimeoutException,
)


# ---------------------------------------------------------
# Existing generate() tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_success(monkeypatch):

    service = OllamaService()

    class MockResponse:

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": "Your order is being processed."
            }

    async def mock_post(self, url, json):
        return MockResponse()

    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        mock_post,
    )

    result = await service.generate(
        "Where is my order?"
    )

    assert result == "Your order is being processed."


# ---------------------------------------------------------
# Streaming Helper
# ---------------------------------------------------------

class MockStreamingResponse:

    def __init__(self, lines):
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        pass

    def raise_for_status(self):
        pass

    async def aiter_lines(self):

        for line in self.lines:
            yield line


# ---------------------------------------------------------
# Test: Streaming Success
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_success(monkeypatch):

    service = OllamaService()

    lines = [
        '{"response":"Your"}',
        '{"response":" order"}',
        '{"response":" is"}',
        '{"response":" being"}',
        '{"response":" processed."}',
        '{"response":"","done":true}',
    ]

    def mock_stream(
        self,
        method,
        url,
        json,
    ):
        return MockStreamingResponse(lines)

    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        mock_stream,
    )

    chunks = []

    async for chunk in service.stream(
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
# Test: Invalid Streaming JSON
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_invalid_json_is_ignored(
    monkeypatch,
):

    service = OllamaService()

    lines = [
        '{"response":"Hello"}',
        "THIS IS NOT JSON",
        '{"response":" world"}',
        '{"response":"","done":true}',
    ]

    def mock_stream(
        self,
        method,
        url,
        json,
    ):
        return MockStreamingResponse(lines)

    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        mock_stream,
    )

    chunks = []

    async for chunk in service.stream(
        "Say hello"
    ):
        chunks.append(chunk)

    assert chunks == [
        "Hello",
        " world",
    ]


# ---------------------------------------------------------
# Test: Empty Stream
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_empty_response(
    monkeypatch,
):

    service = OllamaService()

    lines = [
        '{"response":"","done":true}',
    ]

    def mock_stream(
        self,
        method,
        url,
        json,
    ):
        return MockStreamingResponse(lines)

    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        mock_stream,
    )

    chunks = []

    async for chunk in service.stream(
        "Hello"
    ):
        chunks.append(chunk)

    assert chunks == []


# ---------------------------------------------------------
# Test: Streaming Timeout
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_timeout(
    monkeypatch,
):

    service = OllamaService()

    def mock_stream(
        self,
        method,
        url,
        json,
    ):
        raise httpx.TimeoutException(
            "Request timed out"
        )

    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        mock_stream,
    )

    with pytest.raises(
        LLMTimeoutException
    ):

        async for _ in service.stream(
            "Where is my order?"
        ):
            pass


# ---------------------------------------------------------
# Test: Ollama HTTP Error
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_http_error(
    monkeypatch,
):

    service = OllamaService()

    def mock_stream(
        self,
        method,
        url,
        json,
    ):
        raise httpx.ConnectError(
            "Connection failed"
        )

    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        mock_stream,
    )

    with pytest.raises(LLMException):

        async for _ in service.stream(
            "Where is my order?"
        ):
            pass