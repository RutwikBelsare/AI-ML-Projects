from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Customer support message.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Message cannot be empty or contain only whitespace."
            )

        if any(
            ord(char) < 32 and char not in "\n\t"
            for char in value
        ):
            raise ValueError(
                "Message contains invalid control characters."
            )

        return value