import logging

from openai import AsyncOpenAI

from backend.core.config import get_settings
from backend.core.exceptions import (
    LLMException,
    LLMTimeoutException,
)

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        settings = get_settings()

        if not settings.llm_api_key:
            raise LLMException(
                "LLM API Key is not configured."
            )

        if not settings.llm_model:
            raise LLMException(
                "LLM Model is not configured."
            )

        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            timeout=settings.request_timeout,
        )

        self.model = settings.llm_model

    async def generate(self, message: str) -> str:
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=message,
            )

            if not response.output_text:
                raise LLMException(
                    "The AI service returned an empty response."
                )

            return response.output_text

        except LLMException:
            raise

        except TimeoutError as exc:
            logger.exception("LLM request timed out")
            raise LLMTimeoutException() from exc

        except Exception as exc:
            logger.exception("LLM request failed")
            raise LLMException() from exc