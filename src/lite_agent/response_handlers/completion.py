"""Completion API response handler."""

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litellm import CustomStreamWrapper

from lite_agent.response_handlers.base import ResponseHandler
from lite_agent.stream_handlers import litellm_completion_stream_handler
from lite_agent.types import AgentChunk
from lite_agent.types.events import AssistantMessageEvent, Usage, UsageEvent
from lite_agent.types.messages import AssistantMessageMeta, AssistantTextContent, AssistantToolCall, NewAssistantMessage


class CompletionResponseHandler(ResponseHandler):
    """Handler for Completion API responses."""

    async def _handle_streaming(
        self,
        response: Any,
        record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle streaming completion response."""
        if isinstance(response, CustomStreamWrapper):
            async for chunk in litellm_completion_stream_handler(response, record_to):
                yield chunk
        else:
            msg = "Response is not a CustomStreamWrapper, cannot stream chunks."
            raise TypeError(msg)

    async def _handle_non_streaming(
        self,
        response: Any,
        record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle non-streaming completion response."""
        # Convert completion response to chunks
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            content_items = []

            # Add text content
            if choice.message and choice.message.content:
                content_items.append(AssistantTextContent(text=choice.message.content))

            # Handle tool calls
            if choice.message and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    content_items.append(
                        AssistantToolCall(
                            call_id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )

            # Always yield assistant message, even if content is empty for tool calls
            if choice.message and (content_items or choice.message.tool_calls):
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
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            yield UsageEvent(usage=usage)
