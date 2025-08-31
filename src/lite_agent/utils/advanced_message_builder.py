"""Advanced message builder with fluent interface for complex message construction."""

from datetime import datetime, timezone
from typing import Any

from lite_agent.loggers import logger
from lite_agent.types import (
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    MessageMeta,
    MessageUsage,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    UserImageContent,
    UserTextContent,
)


class FluentMessageBuilder:
    """Fluent interface for building complex messages step by step."""

    def __init__(self):
        self._reset()

    def _reset(self) -> None:
        """Reset builder state."""
        self._message_type = None
        self._content_items = []
        self._meta = None

    def user_message(self) -> "FluentMessageBuilder":
        """Start building a user message."""
        self._reset()
        self._message_type = "user"
        self._meta = MessageMeta()
        logger.debug("Started building user message")
        return self

    def assistant_message(self, model: str | None = None) -> "FluentMessageBuilder":
        """Start building an assistant message."""
        self._reset()
        self._message_type = "assistant"
        self._meta = AssistantMessageMeta(model=model)
        logger.debug(f"Started building assistant message (model: {model})")
        return self

    def system_message(self) -> "FluentMessageBuilder":
        """Start building a system message."""
        self._reset()
        self._message_type = "system"
        self._meta = MessageMeta()
        self._content = ""
        logger.debug("Started building system message")
        return self

    def add_text(self, text: str) -> "FluentMessageBuilder":
        """Add text content."""
        if self._message_type == "user":
            self._content_items.append(UserTextContent(text=text))
        elif self._message_type == "assistant":
            self._content_items.append(AssistantTextContent(text=text))
        elif self._message_type == "system":
            self._content = text
        else:
            msg = "Message type not set. Call user_message(), assistant_message(), or system_message() first."
            raise ValueError(msg)

        logger.debug(f"Added text content (length: {len(text)})")
        return self

    def add_image(self, image_url: str | None = None, file_id: str | None = None, detail: str = "auto") -> "FluentMessageBuilder":
        """Add image content to user message."""
        if self._message_type != "user":
            msg = "Images can only be added to user messages"
            raise ValueError(msg)

        self._content_items.append(UserImageContent(image_url=image_url, file_id=file_id, detail=detail))
        logger.debug(f"Added image content (url: {bool(image_url)}, file_id: {bool(file_id)})")
        return self

    def add_tool_call(self, call_id: str, name: str, arguments: dict[str, Any] | str) -> "FluentMessageBuilder":
        """Add tool call to assistant message."""
        if self._message_type != "assistant":
            msg = "Tool calls can only be added to assistant messages"
            raise ValueError(msg)

        self._content_items.append(AssistantToolCall(call_id=call_id, name=name, arguments=arguments))
        logger.debug(f"Added tool call: {name} (call_id: {call_id})")
        return self

    def add_tool_result(self, call_id: str, output: str, execution_time_ms: int | None = None) -> "FluentMessageBuilder":
        """Add tool call result to assistant message."""
        if self._message_type != "assistant":
            msg = "Tool results can only be added to assistant messages"
            raise ValueError(msg)

        self._content_items.append(AssistantToolCallResult(call_id=call_id, output=output, execution_time_ms=execution_time_ms))
        logger.debug(f"Added tool result for call: {call_id}")
        return self

    def with_timestamp(self, timestamp: datetime | None = None) -> "FluentMessageBuilder":
        """Set message timestamp."""
        if self._meta is None:
            msg = "Message type not set. Call user_message(), assistant_message(), or system_message() first."
            raise ValueError(msg)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self._meta.sent_at = timestamp
        return self

    def with_usage(self, input_tokens: int | None = None, output_tokens: int | None = None) -> "FluentMessageBuilder":
        """Set usage information (assistant messages only)."""
        if self._message_type != "assistant":
            msg = "Usage information can only be set for assistant messages"
            raise ValueError(msg)

        if self._meta is None:
            msg = "Message type not set. Call user_message(), assistant_message(), or system_message() first."
            raise ValueError(msg)

        if not hasattr(self._meta, "usage"):
            self._meta.usage = MessageUsage()

        if self._meta.usage is not None:
            if input_tokens is not None:
                self._meta.usage.input_tokens = input_tokens
            if output_tokens is not None:
                self._meta.usage.output_tokens = output_tokens
            if input_tokens is not None and output_tokens is not None:
                self._meta.usage.total_tokens = input_tokens + output_tokens

        return self

    def with_timing(self, latency_ms: int | None = None, total_time_ms: int | None = None) -> "FluentMessageBuilder":
        """Set timing information (assistant messages only)."""
        if self._message_type != "assistant":
            msg = "Timing information can only be set for assistant messages"
            raise ValueError(msg)

        if self._meta is None:
            msg = "Message type not set. Call user_message(), assistant_message(), or system_message() first."
            raise ValueError(msg)

        if latency_ms is not None:
            self._meta.latency_ms = latency_ms  # type: ignore[attr-defined]
        if total_time_ms is not None:
            self._meta.total_time_ms = total_time_ms  # type: ignore[attr-defined]

        return self

    def build(self) -> NewUserMessage | NewAssistantMessage | NewSystemMessage:
        """Build the final message."""
        if self._message_type == "user":
            message = NewUserMessage(content=self._content_items, meta=self._meta)
        elif self._message_type == "assistant":
            message = NewAssistantMessage(content=self._content_items, meta=self._meta)
        elif self._message_type == "system":
            message = NewSystemMessage(content=self._content, meta=self._meta)
        else:
            msg = "Message type not set"
            raise ValueError(msg)

        logger.debug(f"Built {self._message_type} message with {len(getattr(self, '_content_items', []))} content items")
        return message


class MessageBuilderFactory:
    """Factory for creating common message types quickly."""

    @staticmethod
    def create_simple_user_message(text: str) -> NewUserMessage:
        """Create a simple user text message."""
        return FluentMessageBuilder().user_message().add_text(text).build()

    @staticmethod
    def create_simple_assistant_message(text: str, model: str | None = None) -> NewAssistantMessage:
        """Create a simple assistant text message."""
        return FluentMessageBuilder().assistant_message(model).add_text(text).build()

    @staticmethod
    def create_system_message(text: str) -> NewSystemMessage:
        """Create a system message."""
        return FluentMessageBuilder().system_message().add_text(text).build()

    @staticmethod
    def create_user_message_with_image(text: str, image_url: str) -> NewUserMessage:
        """Create a user message with text and image."""
        return FluentMessageBuilder().user_message().add_text(text).add_image(image_url=image_url).build()

    @staticmethod
    def create_assistant_with_tool_call(text: str, call_id: str, tool_name: str, arguments: dict[str, Any], model: str | None = None) -> NewAssistantMessage:
        """Create an assistant message with text and a tool call."""
        return FluentMessageBuilder().assistant_message(model).add_text(text).add_tool_call(call_id, tool_name, arguments).build()

    @staticmethod
    def create_assistant_with_tool_result(call_id: str, result: str, execution_time_ms: int | None = None, model: str | None = None) -> NewAssistantMessage:
        """Create an assistant message with just a tool result."""
        return FluentMessageBuilder().assistant_message(model).add_tool_result(call_id, result, execution_time_ms).build()
