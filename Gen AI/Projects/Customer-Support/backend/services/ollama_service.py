import json
import logging
from typing import AsyncGenerator

import httpx

from backend.core.config import get_settings
from backend.core.exceptions import (
    LLMException,
    LLMTimeoutException,
)

logger = logging.getLogger(__name__)


class OllamaService:

    def __init__(self):
        settings = get_settings()

        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.timeout = settings.request_timeout

    async def generate(self, message: str) -> str:

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": message,
            "stream": False,
        }

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

            result = data.get("response", "")

            if not result:
                raise LLMException(
                    "The AI service returned an empty response."
                )

            return result

        except LLMException:
            raise

        except httpx.TimeoutException as exc:

            logger.exception(
                "Ollama request timed out"
            )

            raise LLMTimeoutException() from exc

        except httpx.HTTPError as exc:

            logger.exception(
                "Ollama HTTP request failed"
            )

            raise LLMException() from exc

        except Exception as exc:

            logger.exception(
                "Ollama request failed"
            )

            raise LLMException() from exc

    async def stream(
        self,
        message: str,
    ) -> AsyncGenerator[str, None]:

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": message,
            "stream": True,
        }

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                ) as response:

                    response.raise_for_status()

                    async for line in response.aiter_lines():

                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:

                            logger.warning(
                                "Ignoring invalid Ollama stream chunk"
                            )

                            continue

                        chunk = data.get("response", "")

                        if chunk:
                            yield chunk

        except httpx.TimeoutException as exc:

            logger.exception(
                "Ollama streaming request timed out"
            )

            raise LLMTimeoutException() from exc

        except httpx.HTTPError as exc:

            logger.exception(
                "Ollama streaming request failed"
            )

            raise LLMException() from exc

        except Exception as exc:

            logger.exception(
                "Ollama streaming request failed"
            )

            raise LLMException() from exc









    async def chat_with_tools(
    self,
    messages: list[dict],
    tools: list[dict],
) -> dict:
        

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

            return data

        except httpx.TimeoutException as exc:

            logger.exception(
                "Ollama tool request timed out"
            )

            raise LLMTimeoutException() from exc

        except httpx.HTTPError as exc:

            logger.exception(
                "Ollama tool request failed"
            )

            raise LLMException() from exc

        except Exception as exc:

            logger.exception(
                "Ollama tool request failed"
            )

            raise LLMException() from exc





    async def stream_chat_with_tools(
    self,
    messages: list[dict],
    tools: list[dict],
):
        
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                ) as response:

                    response.raise_for_status()

                    async for line in response.aiter_lines():

                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Ignoring invalid Ollama tool stream chunk"
                            )
                            continue

                        yield data

        except httpx.TimeoutException as exc:
            logger.exception(
                "Ollama tool streaming request timed out"
            )
            raise LLMTimeoutException() from exc

        except httpx.HTTPError as exc:
            logger.exception(
                "Ollama tool streaming request failed"
            )
            raise LLMException() from exc

        except Exception as exc:
            logger.exception(
                "Ollama tool streaming request failed"
            )
            raise LLMException() from exc