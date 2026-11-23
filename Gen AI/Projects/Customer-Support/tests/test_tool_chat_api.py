import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.dependencies import get_tool_calling_service


class MockToolCallingService:

    async def process_message(self, message: str) -> str:
        return "Your order ORD1001 is currently shipped."


@pytest.fixture
def client():

    app.dependency_overrides[
        get_tool_calling_service
    ] = lambda: MockToolCallingService()

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_tool_chat_success(client):

    response = client.post(
        "/api/v1/support/tool-chat",
        json={
            "message": "Where is my order ORD1001?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "ORD1001" in data["response"]
    assert "shipped" in data["response"]


def test_tool_chat_validation_empty_message(client):

    response = client.post(
        "/api/v1/support/tool-chat",
        json={
            "message": ""
        },
    )

    assert response.status_code == 422


def test_tool_chat_validation_missing_message(client):

    response = client.post(
        "/api/v1/support/tool-chat",
        json={},
    )

    assert response.status_code == 422


def test_tool_chat_validation_whitespace_message(client):

    response = client.post(
        "/api/v1/support/tool-chat",
        json={
            "message": "     "
        },
    )

    assert response.status_code == 422