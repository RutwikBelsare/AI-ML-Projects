
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Request received when a user submits a research question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Research question submitted by the user",
    )

    thread_id: str | None = Field(
        default=None,
        max_length=128,
        description="Conversation thread ID. Generated if omitted.",
    )

    user_id: str = Field(
        default="local_user",
        min_length=1,
        max_length=128,
        description="User identifier",
    )


class ApprovalRequest(BaseModel):
    """Request to approve or reject pending tool execution."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    user_id: str = Field(
        default="local_user",
        min_length=1,
        max_length=128,
    )

    approved: bool = Field(
        ...,
        description="True to approve, False to reject",
    )


class ChatResponse(BaseModel):
    """Response returned after agent execution."""

    thread_id: str
    user_id: str
    status: str
    final_answer: str = ""
    approval_required: bool = False
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Backend health-check response."""

    status: str
    service: str


class ErrorResponse(BaseModel):
    """Standard API error response."""

    detail: str
