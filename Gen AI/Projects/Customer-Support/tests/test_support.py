from fastapi.testclient import TestClient

from backend.main import app
from backend.services.dependencies import get_support_service


class MockSupportService:

    async def process_message(self, message: str):
        from backend.schemas.responses import SupportResponse

        return SupportResponse(
            category="shipping",
            intent="order_status",
            sentiment="neutral",
            priority="medium",
            response="Your order is currently being processed.",
            requires_human=False,
        )


def override_support_service():
    return MockSupportService()


app.dependency_overrides[get_support_service] = (
    override_support_service
)

client = TestClient(app)


def test_chat_success():
    response = client.post(
        "/api/v1/support/chat",
        json={
            "message": "Where is my order?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert (
        body["data"]["response"]
        == "Your order is currently being processed."
    )
    assert body["data"]["category"] == "shipping"
    assert body["data"]["intent"] == "order_status"


def test_chat_empty_message():
    response = client.post(
        "/api/v1/support/chat",
        json={
            "message": ""
        },
    )

    assert response.status_code == 422


def test_chat_whitespace_message():
    response = client.post(
        "/api/v1/support/chat",
        json={
            "message": "   "
        },
    )

    assert response.status_code == 422


def test_chat_message_too_long():
    response = client.post(
        "/api/v1/support/chat",
        json={
            "message": "a" * 4001
        },
    )

    assert response.status_code == 422