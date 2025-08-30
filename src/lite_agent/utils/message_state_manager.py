"""Message state management utilities to prevent race conditions."""

import asyncio
from typing import Any

from lite_agent.loggers import logger
from lite_agent.types import (
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    NewAssistantMessage,
)


class MessageStateManager:
    """Thread-safe manager for assistant message state during streaming."""

    def __init__(self):
        self._current_message: NewAssistantMessage | None = None
        self._lock = asyncio.Lock()
        self._finalized = False

    async def start_message(self, content: str = "", meta: AssistantMessageMeta | None = None) -> None:
        """Start a new assistant message safely."""
        async with self._lock:
            if self._current_message is not None:
                logger.warning("Starting new message while previous message not finalized")

            if meta is None:
                meta = AssistantMessageMeta()

            self._current_message = NewAssistantMessage(
                content=[AssistantTextContent(text=content)] if content else [],
                meta=meta,
            )
            self._finalized = False
            logger.debug("Started new assistant message")

    async def ensure_message_exists(self) -> NewAssistantMessage:
        """Ensure current message exists, create if necessary."""
        async with self._lock:
            if self._current_message is None:
                await self._start_message_internal()
            if self._current_message is None:
                raise RuntimeError("Failed to create current assistant message")
            return self._current_message

    async def _start_message_internal(self) -> None:
        """Internal method to start message without lock (already locked)."""
        meta = AssistantMessageMeta()
        self._current_message = NewAssistantMessage(
            content=[],
            meta=meta,
        )
        self._finalized = False

    async def add_text_delta(self, delta: str) -> None:
        """Add text delta to current message safely."""
        if not delta:
            return

        async with self._lock:
            if self._current_message is None:
                logger.debug("Creating new message for text delta")
                await self._start_message_internal()

            if self._current_message is None:
                raise RuntimeError("Failed to ensure current message exists")

            # Find existing text content or create new one
            for item in self._current_message.content:
                if item.type == "text":
                    logger.debug(f"Appending text delta (length: {len(delta)})")
                    item.text += delta
                    return

            # No text content found, add new one
            logger.debug(f"Adding new text content with delta (length: {len(delta)})")
            self._current_message.content.append(AssistantTextContent(text=delta))

    async def add_tool_call(self, tool_call: AssistantToolCall) -> None:
        """Add tool call to current message safely."""
        async with self._lock:
            if self._current_message is None:
                await self._start_message_internal()

            if self._current_message is None:
                raise RuntimeError("Failed to ensure current message exists")

            self._current_message.content.append(tool_call)
            logger.debug(f"Added tool call: {tool_call.name}")

    async def add_tool_result(self, result: AssistantToolCallResult) -> None:
        """Add tool call result to current message safely."""
        async with self._lock:
            if self._current_message is None:
                await self._start_message_internal()

            if self._current_message is None:
                raise RuntimeError("Failed to ensure current message exists")

            self._current_message.content.append(result)
            logger.debug(f"Added tool result for call: {result.call_id}")

    async def update_meta(self, **kwargs: Any) -> None:
        """Update message metadata safely."""
        async with self._lock:
            if self._current_message is None:
                return

            for key, value in kwargs.items():
                if hasattr(self._current_message.meta, key):
                    setattr(self._current_message.meta, key, value)

    async def get_current_message(self) -> NewAssistantMessage | None:
        """Get current message safely."""
        async with self._lock:
            return self._current_message

    async def finalize_message(self) -> NewAssistantMessage | None:
        """Finalize and return current message, reset state."""
        async with self._lock:
            if self._current_message is None or self._finalized:
                return None

            finalized_message = self._current_message
            self._current_message = None
            self._finalized = True
            logger.debug("Finalized assistant message")
            return finalized_message

    async def reset(self) -> None:
        """Reset state manager."""
        async with self._lock:
            self._current_message = None
            self._finalized = False
            logger.debug("Reset message state manager")

    @property
    def has_current_message(self) -> bool:
        """Check if there's a current message (non-blocking check)."""
        return self._current_message is not None

    @property
    def is_finalized(self) -> bool:
        """Check if current message is finalized (non-blocking check)."""
        return self._finalized
