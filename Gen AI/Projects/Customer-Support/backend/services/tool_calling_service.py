import json
import logging

from backend.core.exceptions import LLMException
from backend.services.tool_service import execute_tool
from backend.tools.tool_registry import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)


class ToolCallingService:

    def __init__(self, llm_service):
        self.llm_service = llm_service

    # =====================================================
    # NON-STREAMING TOOL CALLING
    # =====================================================

    async def process_message(
        self,
        message: str,
    ) -> str:

        messages = [
            {
                "role": "user",
                "content": message,
            }
        ]

        # -------------------------------------------------
        # 1. Ask the LLM whether a tool is required
        # -------------------------------------------------

        response = await self.llm_service.chat_with_tools(
            messages=messages,
            tools=AVAILABLE_TOOLS,
        )

        assistant_message = response.get(
            "message",
            {},
        )

        tool_calls = assistant_message.get(
            "tool_calls",
            [],
        )

        # -------------------------------------------------
        # 2. No tool required
        # -------------------------------------------------

        if not tool_calls:

            content = assistant_message.get(
                "content",
                "",
            )

            if not content:

                raise LLMException(
                    "The AI service returned an empty response."
                )

            return content

        # -------------------------------------------------
        # 3. Add assistant tool-call message
        # -------------------------------------------------

        messages.append(
            assistant_message
        )

        # -------------------------------------------------
        # 4. Execute tools
        # -------------------------------------------------

        for tool_call in tool_calls:

            function = tool_call.get(
                "function",
                {},
            )

            tool_name = function.get(
                "name"
            )

            arguments = function.get(
                "arguments",
                {},
            )

            # Ollama may return arguments as a
            # dictionary or JSON string.
            if isinstance(arguments, str):

                try:

                    arguments = json.loads(
                        arguments
                    )

                except json.JSONDecodeError as exc:

                    raise LLMException(
                        "The AI service returned invalid tool arguments."
                    ) from exc

            try:

                result = execute_tool(
                    tool_name,
                    arguments,
                )

            except Exception as exc:

                logger.exception(
                    "Tool execution failed | tool=%s",
                    tool_name,
                )

                raise LLMException(
                    "The requested support operation "
                    "could not be completed."
                ) from exc

            # -------------------------------------------------
            # 5. Add tool result to conversation
            # -------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(result),
                }
            )

        # -------------------------------------------------
        # 6. Ask LLM for final response
        # -------------------------------------------------

        final_response = await self.llm_service.chat_with_tools(
            messages=messages,
            tools=AVAILABLE_TOOLS,
        )

        final_message = final_response.get(
            "message",
            {},
        )

        final_content = final_message.get(
            "content",
            "",
        )

        if not final_content:

            raise LLMException(
                "The AI service returned an empty final response."
            )

        return final_content

    # =====================================================
    # STREAMING TOOL CALLING
    # =====================================================

    async def stream_message(
        self,
        message: str,
    ):
        """
        Handles tool calling and streams only the
        final customer-facing response.

        Flow:

        1. Non-streaming LLM call determines whether
           a tool is required.

        2. If no tool is required:
           make a streaming LLM call.

        3. If a tool is required:
           execute the tool.

        4. Send the tool result back to the LLM.

        5. Stream the final customer-facing response.
        """

        messages = [
            {
                "role": "user",
                "content": message,
            }
        ]

        # =================================================
        # STEP 1 — Determine whether a tool is required
        # =================================================

        response = await self.llm_service.chat_with_tools(
            messages=messages,
            tools=AVAILABLE_TOOLS,
        )

        assistant_message = response.get(
            "message",
            {},
        )

        tool_calls = assistant_message.get(
            "tool_calls",
            [],
        )

        # =================================================
        # CASE 1 — No tool required
        # =================================================

        if not tool_calls:

            # We already have an LLM response, but we want
            # genuine streaming for the customer.
            #
            # Therefore we make a streaming call using the
            # original user message.

            async for chunk in (
                self.llm_service.stream_chat_with_tools(
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                )
            ):

                message_data = chunk.get(
                    "message",
                    {},
                )

                chunk_content = message_data.get(
                    "content",
                    "",
                )

                if chunk_content:

                    yield chunk_content

            return

        # =================================================
        # CASE 2 — Tool required
        # =================================================

        # Add the assistant's tool-call message.
        messages.append(
            assistant_message
        )

        # =================================================
        # STEP 2 — Execute tools
        # =================================================

        for tool_call in tool_calls:

            function = tool_call.get(
                "function",
                {},
            )

            tool_name = function.get(
                "name"
            )

            arguments = function.get(
                "arguments",
                {},
            )

            # -------------------------------------------------
            # Convert JSON string → dictionary
            # -------------------------------------------------

            if isinstance(arguments, str):

                try:

                    arguments = json.loads(
                        arguments
                    )

                except json.JSONDecodeError as exc:

                    raise LLMException(
                        "The AI service returned invalid tool arguments."
                    ) from exc

            # -------------------------------------------------
            # Execute allow-listed tool
            # -------------------------------------------------

            try:

                result = execute_tool(
                    tool_name,
                    arguments,
                )

            except Exception as exc:

                logger.exception(
                    "Tool execution failed | tool=%s",
                    tool_name,
                )

                raise LLMException(
                    "The requested support operation "
                    "could not be completed."
                ) from exc

            # -------------------------------------------------
            # Add tool result to conversation
            # -------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(result),
                }
            )

        # =================================================
        # STEP 3 — Stream final LLM response
        # =================================================

        final_content_received = False

        async for chunk in (
            self.llm_service.stream_chat_with_tools(
                messages=messages,
                tools=AVAILABLE_TOOLS,
            )
        ):

            message_data = chunk.get(
                "message",
                {},
            )

            chunk_content = message_data.get(
                "content",
                "",
            )

            if chunk_content:

                final_content_received = True

                # This is genuine customer-facing
                # streaming.
                yield chunk_content

        # =================================================
        # STEP 4 — Validate final response
        # =================================================

        if not final_content_received:

            raise LLMException(
                "The AI service returned an empty final response."
            )