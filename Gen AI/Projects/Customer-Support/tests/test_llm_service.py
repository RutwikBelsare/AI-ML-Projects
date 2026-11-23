import pytest

from backend.core.config import get_settings
from backend.core.exceptions import LLMException, LLMTimeoutException
from backend.services.llm_service import LLMService
from unittest.mock import AsyncMock, MagicMock


def test_llm_service_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    get_settings.cache_clear()

    with pytest.raises(LLMException) as exc_info:
        LLMService()

    assert exc_info.value.code == "LLM_ERROR"
    assert "api key" in exc_info.value.message.lower()


def test_llm_service_requires_model(monkeypatch):
    monkeypatch.setenv(
        "LLM_API_KEY",
        "test-api-key",
    )
    monkeypatch.setenv("LLM_MODEL", "")

    get_settings.cache_clear()

    with pytest.raises(LLMException) as exc_info:
        LLMService()

    assert exc_info.value.code == "LLM_ERROR"
    assert "model" in exc_info.value.message.lower()

@pytest.mark.asyncio
async def test_llm_generate_success(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    get_settings.cache_clear()

    service = LLMService()

    mock_response = MagicMock()
    mock_response.output_text = "Your order is on the way."

    service.client.responses.create = AsyncMock(
        return_value=mock_response
    )

    result = await service.generate(
        "Where is my order?"
    )

    assert result == "Your order is on the way."

    service.client.responses.create.assert_awaited_once_with(
        model="test-model",
        input="Where is my order?",
    )


@pytest.mark.asyncio
async def test_llm_generate_empty_response(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    get_settings.cache_clear()

    service = LLMService()

    mock_response = MagicMock()
    mock_response.output_text = ""

    service.client.responses.create = AsyncMock(
        return_value=mock_response
    )

    with pytest.raises(LLMException) as exc_info:
        await service.generate(
            "Where is my order?"
        )

    assert exc_info.value.code == "LLM_ERROR"
    assert "empty response" in exc_info.value.message.lower()



@pytest.mark.asyncio
async def test_llm_generate_timeout(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    get_settings.cache_clear()

    service = LLMService()

    service.client.responses.create = AsyncMock(
        side_effect=TimeoutError()
    )

    with pytest.raises(LLMTimeoutException) as exc_info:
        await service.generate(
            "Where is my order?"
        )

    assert exc_info.value.code == "LLM_TIMEOUT"


@pytest.mark.asyncio
async def test_llm_generate_provider_failure(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    get_settings.cache_clear()

    service = LLMService()

    service.client.responses.create = AsyncMock(
        side_effect=Exception("Provider unavailable")
    )

    with pytest.raises(LLMException) as exc_info:
        await service.generate(
            "Where is my order?"
        )

    assert exc_info.value.code == "LLM_ERROR"



