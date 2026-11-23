import json
from typing import Iterator

import httpx


class APIClientError(Exception):
    """Raised when the backend API returns an error."""


class APIClient:

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, message: str) -> dict:
        """
        Send a normal non-streaming chat request.
        """

        try:
            response = httpx.post(
                f"{self.base_url}/api/v1/support/chat",
                json={"message": message},
                timeout=self.timeout,
            )

            if response.status_code >= 400:
                self._raise_api_error(response)

            return response.json()

        except httpx.TimeoutException as exc:
            raise APIClientError(
                "The backend request timed out."
            ) from exc

        except httpx.ConnectError as exc:
            raise APIClientError(
                "Unable to connect to the backend."
            ) from exc

        except httpx.HTTPError as exc:
            raise APIClientError(
                "A backend communication error occurred."
            ) from exc

        except ValueError as exc:
            raise APIClientError(
                "The backend returned an invalid response."
            ) from exc

    def stream_chat(
        self,
        message: str,
    ) -> Iterator[str]:
        """
        Stream the AI response from the backend.
        """

        try:

            with httpx.stream(
                "POST",
                f"{self.base_url}/api/v1/support/chat/stream",
                json={"message": message},
                timeout=self.timeout,
            ) as response:

                if response.status_code >= 400:
                    self._raise_api_error(response)

                for chunk in response.iter_text():

                    if chunk:
                        yield chunk

        except httpx.TimeoutException as exc:

            raise APIClientError(
                "The streaming request timed out."
            ) from exc

        except httpx.ConnectError as exc:

            raise APIClientError(
                "Unable to connect to the backend."
            ) from exc

        except httpx.HTTPError as exc:

            raise APIClientError(
                "A backend communication error occurred."
            ) from exc

    @staticmethod
    def _raise_api_error(
        response: httpx.Response,
    ) -> None:

        try:
            data = response.json()

            error = data.get("error", {})

            message = error.get(
                "message",
                "The backend returned an error.",
            )

        except (ValueError, json.JSONDecodeError):
            message = (
                "The backend returned an unexpected error."
            )

        raise APIClientError(message)