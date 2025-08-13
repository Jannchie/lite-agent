"""Responses API response handler."""
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lite_agent.response_handlers.base import ResponseHandler
from lite_agent.stream_handlers import litellm_response_stream_handler
from lite_agent.types import AgentChunk
from lite_agent.types.events import AssistantMessageEvent
from lite_agent.types.messages import AssistantMessageMeta, AssistantTextContent, NewAssistantMessage


class ResponsesAPIHandler(ResponseHandler):
    """Handler for Responses API responses."""

    async def _handle_streaming(
        self, response: Any, record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle streaming responses API response."""
        async for chunk in litellm_response_stream_handler(response, record_to):
            yield chunk

    async def _handle_non_streaming(
        self, response: Any, record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle non-streaming responses API response."""
        # Convert ResponsesAPIResponse to chunks
        if hasattr(response, "output") and response.output:
            for output_message in response.output:
                if hasattr(output_message, "content") and output_message.content:
                    content_text = ""
                    for content_item in output_message.content:
                        if hasattr(content_item, "text"):
                            content_text += content_item.text

                    if content_text:
                        message = NewAssistantMessage(
                            content=[AssistantTextContent(text=content_text)],
                            meta=AssistantMessageMeta(sent_at=datetime.now(timezone.utc)),
                        )
                        yield AssistantMessageEvent(message=message)
