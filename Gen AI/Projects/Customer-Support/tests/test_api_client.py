import httpx
import pytest

from frontend.api_client import (
    APIClient,
    APIClientError,
)


# ---------------------------------------------------------
# Mock Streaming Response
# ---------------------------------------------------------

class MockStreamingResponse:

    def __init__(
        self,
        status_code=200,
        chunks=None,
        json_data=None,
    ):
        self.status_code = status_code
        self.chunks = chunks or []
        self.json_data = json_data or {}

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        pass

    def iter_text(self):

        for chunk in self.chunks:
            yield chunk

    def json(self):
        return self.json_data


# ---------------------------------------------------------
# Test: Normal Chat Success
# ---------------------------------------------------------

def test_chat_success(monkeypatch):

    client = APIClient()

    class MockResponse:

        status_code = 200

        def json(self):
            return {
                "success": True,
                "data": {
                    "category": "order",
                    "intent": "order_status",
                    "sentiment": "neutral",
                    "priority": "medium",
                    "response": "Your order is being processed.",
                    "requires_human": False,
                },
            }

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    result = client.chat(
        "Where is my order?"
    )

    assert result["success"] is True
    assert (
        result["data"]["intent"]
        == "order_status"
    )


# ---------------------------------------------------------
# Test: Backend Connection Error
# ---------------------------------------------------------

def test_chat_connection_error(monkeypatch):

    client = APIClient()

    def mock_post(*args, **kwargs):
        raise httpx.ConnectError(
            "Connection refused"
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    with pytest.raises(APIClientError) as exc_info:

        client.chat(
            "Where is my order?"
        )

    assert (
        str(exc_info.value)
        == "Unable to connect to the backend."
    )


# ---------------------------------------------------------
# Test: Backend Timeout
# ---------------------------------------------------------

def test_chat_timeout(monkeypatch):

    client = APIClient()

    def mock_post(*args, **kwargs):
        raise httpx.TimeoutException(
            "Request timed out"
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    with pytest.raises(APIClientError) as exc_info:

        client.chat(
            "Where is my order?"
        )

    assert (
        str(exc_info.value)
        == "The backend request timed out."
    )


# ---------------------------------------------------------
# Test: Invalid Backend Response
# ---------------------------------------------------------

def test_chat_invalid_json(monkeypatch):

    client = APIClient()

    class MockResponse:

        status_code = 200

        def json(self):
            raise ValueError(
                "Invalid JSON"
            )

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    with pytest.raises(APIClientError) as exc_info:

        client.chat(
            "Where is my order?"
        )

    assert (
        str(exc_info.value)
        == "The backend returned an invalid response."
    )


# ---------------------------------------------------------
# Test: Backend API Error
# ---------------------------------------------------------

def test_chat_api_error(monkeypatch):

    client = APIClient()

    class MockResponse:

        status_code = 502

        def json(self):
            return {
                "error": {
                    "code": "LLM_ERROR",
                    "message": "The AI service is temporarily unavailable.",
                }
            }

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    with pytest.raises(APIClientError) as exc_info:

        client.chat(
            "Where is my order?"
        )

    assert (
        str(exc_info.value)
        == "The AI service is temporarily unavailable."
    )


# ---------------------------------------------------------
# Test: Streaming Success
# ---------------------------------------------------------

def test_stream_chat_success(monkeypatch):

    client = APIClient()

    response = MockStreamingResponse(
        status_code=200,
        chunks=[
            "Your",
            " order",
            " is",
            " being",
            " processed.",
        ],
    )

    def mock_stream(*args, **kwargs):
        return response

    monkeypatch.setattr(
        httpx,
        "stream",
        mock_stream,
    )

    chunks = list(
        client.stream_chat(
            "Where is my order?"
        )
    )

    assert chunks == [
        "Your",
        " order",
        " is",
        " being",
        " processed.",
    ]


# ---------------------------------------------------------
# Test: Streaming Connection Error
# ---------------------------------------------------------

def test_stream_chat_connection_error(
    monkeypatch,
):

    client = APIClient()

    def mock_stream(*args, **kwargs):
        raise httpx.ConnectError(
            "Connection refused"
        )

    monkeypatch.setattr(
        httpx,
        "stream",
        mock_stream,
    )

    with pytest.raises(APIClientError) as exc_info:

        list(
            client.stream_chat(
                "Where is my order?"
            )
        )

    assert (
        str(exc_info.value)
        == "Unable to connect to the backend."
    )


# ---------------------------------------------------------
# Test: Streaming Timeout
# ---------------------------------------------------------

def test_stream_chat_timeout(
    monkeypatch,
):

    client = APIClient()

    def mock_stream(*args, **kwargs):
        raise httpx.TimeoutException(
            "Request timed out"
        )

    monkeypatch.setattr(
        httpx,
        "stream",
        mock_stream,
    )

    with pytest.raises(APIClientError) as exc_info:

        list(
            client.stream_chat(
                "Where is my order?"
            )
        )

    assert (
        str(exc_info.value)
        == "The streaming request timed out."
    )