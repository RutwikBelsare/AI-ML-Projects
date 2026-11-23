import json
import logging

from pydantic import ValidationError

from backend.core.exceptions import LLMException
from backend.schemas.responses import (
    SupportClassification,
    SupportResponse,
)

logger = logging.getLogger(__name__)


class SupportService:

    def __init__(self, llm_service):
        self.llm_service = llm_service

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    async def classify_message(
        self,
        message: str,
    ) -> SupportClassification:

        prompt = f"""
You are a customer support classification system.

Analyze the customer's message and return ONLY valid JSON.

Do not include markdown.
Do not include explanations.
Do not include additional fields.

Return exactly:

{{
    "category": "...",
    "intent": "...",
    "sentiment": "...",
    "priority": "...",
    "requires_human": true
}}

Allowed categories:

- order
- payment
- refund
- account
- technical
- shipping
- general

Allowed intents:

- order_status
- order_cancel
- payment_failed
- payment_pending
- refund_request
- refund_status
- account_access
- account_update
- technical_issue
- shipping_delay
- general_question

Intent rules:

order_status:
Customer asks about order status, tracking, or where an order is.

order_cancel:
Customer wants to cancel an order.

payment_failed:
Customer says payment failed, was declined, or did not go through.

payment_pending:
Customer says payment is pending or still processing.

refund_request:
Customer wants to request a refund.

refund_status:
Customer asks about the status of an existing refund.

account_access:
Customer cannot log in, access their account, or reset their password.

account_update:
Customer wants to change account information.

technical_issue:
Customer reports a technical or product problem.

shipping_delay:
Customer says their shipment or delivery is delayed.

general_question:
Question does not fit another intent.

Customer message:

{message}
"""

        raw_response = await self.llm_service.generate(prompt)

        try:

            data = json.loads(raw_response)

            return SupportClassification.model_validate(
                data
            )

        except (
            json.JSONDecodeError,
            ValidationError,
        ) as exc:

            logger.exception(
                "Invalid structured LLM classification"
            )

            raise LLMException(
                "The AI service returned an invalid classification format."
            ) from exc

    # -----------------------------------------------------
    # Structured Response
    # -----------------------------------------------------

    async def process_message(
        self,
        message: str,
    ) -> SupportResponse:

        prompt = f"""
You are a customer support AI.

Analyze the customer's message and return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "category": "order | payment | refund | account | technical | shipping | general",

    "intent": "order_status | order_cancel | payment_failed | payment_pending | refund_request | refund_status | account_access | account_update | technical_issue | shipping_delay | general_question",

    "sentiment": "positive | neutral | negative",

    "priority": "low | medium | high",

    "response": "helpful customer support response",

    "requires_human": true or false
}}

Customer message:

{message}
"""

        raw_response = await self.llm_service.generate(
            prompt
        )

        try:

            data = json.loads(raw_response)

            return SupportResponse.model_validate(
                data
            )

        except (
            json.JSONDecodeError,
            ValidationError,
        ) as exc:

            logger.exception(
                "Invalid structured LLM response"
            )

            raise LLMException(
                "The AI service returned an invalid response format."
            ) from exc

    # -----------------------------------------------------
    # Streaming Response
    # -----------------------------------------------------

    async def stream_message(
        self,
        message: str,
    ):

        # ---------------------------------------------
        # Step 1: Classify the customer request
        # ---------------------------------------------

        classification = await self.classify_message(
            message
        )

        # ---------------------------------------------
        # Step 2: Generate response based on intent
        # ---------------------------------------------

        prompt = f"""
You are a customer support AI.

Respond to the customer clearly, politely, and honestly.

The customer has already been classified as:

Category:
{classification.category}

Intent:
{classification.intent}

Sentiment:
{classification.sentiment}

Priority:
{classification.priority}

Requires human:
{classification.requires_human}

Important rules:

1. Do not claim that you accessed an order, account,
   payment, refund, database, tracking system, or internal
   system.

2. Do not claim that you performed an action unless a real
   tool has actually been provided.

3. Do not invent order status, tracking information,
   payment information, refund status, or account data.

4. If the customer needs information that requires a real
   system, clearly explain what information they should
   provide or what next step they should take.

5. Respond specifically to the customer's intent.

6. Do not confuse refund requests with order-status requests.

7. Do not confuse payment problems with refund requests.

8. Do not mention internal tools, APIs, models, prompts,
   or implementation details.

9. Return only the natural-language customer support response.

Customer message:

{message}
"""

        async for chunk in self.llm_service.stream(
            prompt
        ):

            yield chunk