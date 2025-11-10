"""Responses API response handler."""

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lite_agent.response_handlers.base import ResponseHandler
from lite_agent.stream_handlers import openai_response_stream_handler
from lite_agent.types import AgentChunk
from lite_agent.types.events import AssistantMessageEvent, Usage, UsageEvent
from lite_agent.types.messages import AssistantMessageMeta, AssistantTextContent, AssistantToolCall, NewAssistantMessage


class ResponsesAPIHandler(ResponseHandler):
    """Handler for Responses API responses."""

    async def _handle_streaming(
        self,
        response: Any,  # noqa: ANN401
        record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle streaming responses API response."""
        if not hasattr(response, "__aiter__"):
            msg = "Response does not support async iteration, cannot stream chunks."
            raise TypeError(msg)

        async for chunk in openai_response_stream_handler(response, record_to):
            yield chunk

    async def _handle_non_streaming(
        self,
        response: Any,  # noqa: ANN401
        record_to: Path | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle non-streaming responses API response."""
        # Convert ResponsesAPIResponse to chunks
        if hasattr(response, "output") and response.output:
            content_items = []

            for output_item in response.output:
                # Handle function tool calls
                if hasattr(output_item, "type") and output_item.type == "function_call":
                    content_items.append(
                        AssistantToolCall(
                            call_id=output_item.call_id,
                            name=output_item.name,
                            arguments=output_item.arguments,
                        ),
                    )
                # Handle text content (if exists)
                elif hasattr(output_item, "content") and output_item.content:
                    content_text = ""
                    for content_item in output_item.content:
                        if hasattr(content_item, "text"):
                            content_text += content_item.text

                    if content_text:
                        content_items.append(AssistantTextContent(text=content_text))

            # Create assistant message if we have any content
            if content_items:
                # Extract model information from response
                model_name = getattr(response, "model", None)
                message = NewAssistantMessage(
                    content=content_items,
                    meta=AssistantMessageMeta(
                        sent_at=datetime.now(timezone.utc),
                        model=model_name,
                    ),
                )
                yield AssistantMessageEvent(message=message)

        # Yield usage information if available
        if hasattr(response, "usage") and response.usage:
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            yield UsageEvent(usage=usage)
