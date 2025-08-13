"""Completion API response handler."""
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litellm import CustomStreamWrapper

from lite_agent.response_handlers.base import ResponseHandler
from lite_agent.stream_handlers import litellm_completion_stream_handler
from lite_agent.types import AgentChunk
from lite_agent.types.events import AssistantMessageEvent
from lite_agent.types.messages import AssistantMessageMeta, AssistantTextContent, NewAssistantMessage


class CompletionResponseHandler(ResponseHandler):
    """Handler for Completion API responses."""

    async def _handle_streaming(
        self, response: Any, record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle streaming completion response."""
        if isinstance(response, CustomStreamWrapper):
            async for chunk in litellm_completion_stream_handler(response, record_to):
                yield chunk
        else:
            msg = "Response is not a CustomStreamWrapper, cannot stream chunks."
            raise TypeError(msg)

    async def _handle_non_streaming(
        self, response: Any, record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle non-streaming completion response."""
        # Convert completion response to chunks
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            content_items = []

            # Add text content
            if choice.message and choice.message.content:
                content_items.append(AssistantTextContent(text=choice.message.content))

            # TODO: Handle tool calls in the future

            if content_items:
                message = NewAssistantMessage(
                    content=content_items,
                    meta=AssistantMessageMeta(sent_at=datetime.now(timezone.utc)),
                )
                yield AssistantMessageEvent(message=message)
