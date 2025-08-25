"""Base response handler for unified streaming and non-streaming response processing."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from lite_agent.types import AgentChunk


class ResponseHandler(ABC):
    """Base class for handling both streaming and non-streaming responses."""

    async def handle(
        self,
        response: Any,  # noqa: ANN401
        *,
        streaming: bool,
        record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle a response in either streaming or non-streaming mode.

        Args:
            response: The LLM response object
            streaming: Whether to process as streaming or non-streaming
            record_to: Optional file path to record the conversation

        Yields:
            AgentChunk: Processed chunks from the response
        """
        if streaming:
            stream = self._handle_streaming(response, record_to)
            async for chunk in stream:
                yield chunk
        else:
            stream = self._handle_non_streaming(response, record_to)
            async for chunk in stream:
                yield chunk

    @abstractmethod
    def _handle_streaming(
        self,
        response: Any,  # noqa: ANN401
        record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle streaming response."""

    @abstractmethod
    def _handle_non_streaming(
        self,
        response: Any,  # noqa: ANN401
        record_to: Path | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Handle non-streaming response."""
