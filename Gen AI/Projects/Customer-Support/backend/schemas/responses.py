from typing import Literal

from pydantic import BaseModel, Field


class SupportClassification(BaseModel):

    category: Literal[
        "order",
        "payment",
        "refund",
        "account",
        "technical",
        "shipping",
        "general",
    ]

    intent: Literal[
        "order_status",
        "order_cancel",
        "payment_failed",
        "payment_pending",
        "refund_request",
        "refund_status",
        "account_access",
        "account_update",
        "technical_issue",
        "shipping_delay",
        "general_question",
    ]

    sentiment: Literal[
        "positive",
        "neutral",
        "negative",
    ]

    priority: Literal[
        "low",
        "medium",
        "high",
    ]

    requires_human: bool


class SupportResponse(SupportClassification):

    response: str = Field(
        min_length=1,
        max_length=5000,
    )


class ChatResponse(BaseModel):
    success: bool
    data: SupportResponse