import pytest
from pydantic import ValidationError
from backend.schemas.requests import ChatRequest

def test_valid_chat_request():
    request = ChatRequest(message="Where is my order?")

    assert request.message == "Where is my order?"

def test_message_is_trimmed():
    request = ChatRequest(message = " Where is my order? ")
    assert request.message == "Where is my order?"

def test_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(mesage="")

def test_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(message = " ")

def test_message_too_long():
    with pytest.raises(ValidationError):
        ChatRequest(message = "a" * 4001)

def test_message_max_length_allowed():
    request = ChatRequest(message= "a" * 4000)
    assert len(request.message) == 4000

def test_invalid_control_character():
    with pytest.raises(ValidationError):
        ChatRequest(message = "Hello\x00World")

























