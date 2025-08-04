from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field, model_validator


# Base metadata type
class MessageMeta(BaseModel):
    """Base metadata for all message types"""

    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BasicMessageMeta(MessageMeta):
    """Basic metadata for user messages and function calls"""

    execution_time_ms: int | None = None


class LLMResponseMeta(MessageMeta):
    """Metadata for LLM responses, includes performance metrics"""

    latency_ms: int | None = None
    output_time_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


# New unified metadata types


class MessageUsage(BaseModel):
    """Token usage statistics for messages"""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class AssistantMessageMeta(MessageMeta):
    """Enhanced metadata for assistant messages"""

    model: str | None = None
    usage: MessageUsage | None = None
    total_time_ms: int | None = None
    latency_ms: int | None = None
    output_time_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ResponseInputImageDict(TypedDict):
    detail: NotRequired[Literal["low", "high", "auto"]]
    type: Literal["input_image"]
    file_id: str | None
    image_url: str | None


class ResponseInputTextDict(TypedDict):
    text: str
    type: Literal["input_text"]


# TypedDict definitions for better type hints
class UserMessageDict(TypedDict):
    role: Literal["user"]
    content: str | Sequence[ResponseInputTextDict | ResponseInputImageDict]


class AssistantMessageDict(TypedDict):
    role: Literal["assistant"]
    content: str


class SystemMessageDict(TypedDict):
    role: Literal["system"]
    content: str


class FunctionCallDict(TypedDict):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str
    content: str


class FunctionCallOutputDict(TypedDict):
    type: Literal["function_call_output"]
    call_id: str
    output: str


# Union type for all supported message dictionary formats
MessageDict = UserMessageDict | AssistantMessageDict | SystemMessageDict | FunctionCallDict | FunctionCallOutputDict


# New structured message content types
class UserTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class UserImageContent(BaseModel):
    type: Literal["image"] = "image"
    image_url: str | None = None
    file_id: str | None = None
    detail: Literal["low", "high", "auto"] = "auto"

    @model_validator(mode="after")
    def validate_image_source(self) -> "UserImageContent":
        if not self.file_id and not self.image_url:
            msg = "UserImageContent must have either file_id or image_url"
            raise ValueError(msg)
        return self


class UserFileContent(BaseModel):
    type: Literal["file"] = "file"
    file_id: str
    file_name: str | None = None


UserMessageContent = UserTextContent | UserImageContent | UserFileContent


class AssistantTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AssistantToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any] | str


class AssistantToolCallResult(BaseModel):
    type: Literal["tool_call_result"] = "tool_call_result"
    call_id: str
    output: str
    execution_time_ms: int | None = None


AssistantMessageContent = AssistantTextContent | AssistantToolCall | AssistantToolCallResult


# New structured message types
class NewUserMessage(BaseModel):
    """User message with structured content support"""

    role: Literal["user"] = "user"
    content: list[UserMessageContent]
    meta: MessageMeta = Field(default_factory=MessageMeta)


class NewSystemMessage(BaseModel):
    """System message"""

    role: Literal["system"] = "system"
    content: str
    meta: MessageMeta = Field(default_factory=MessageMeta)


class NewAssistantMessage(BaseModel):
    """Assistant message with structured content and metadata"""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantMessageContent]
    meta: AssistantMessageMeta = Field(default_factory=AssistantMessageMeta)


# Union type for new structured messages
NewMessage = NewUserMessage | NewSystemMessage | NewAssistantMessage
NewMessages = Sequence[NewMessage]


# Response API format input types
class ResponseInputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class ResponseInputImage(BaseModel):
    detail: Literal["low", "high", "auto"] = "auto"
    type: Literal["input_image"] = "input_image"
    file_id: str | None = None
    image_url: str | None = None

    @model_validator(mode="after")
    def validate_image_source(self) -> "ResponseInputImage":
        """Ensure at least one of file_id or image_url is provided."""
        if not self.file_id and not self.image_url:
            msg = "ResponseInputImage must have either file_id or image_url"
            raise ValueError(msg)
        return self


# Compatibility types for old completion API format
class UserMessageContentItemText(BaseModel):
    type: Literal["text"]
    text: str


class UserMessageContentItemImageURLImageURL(BaseModel):
    url: str


class UserMessageContentItemImageURL(BaseModel):
    type: Literal["image_url"]
    image_url: UserMessageContentItemImageURLImageURL


# Legacy compatibility wrapper classes
class AgentUserMessage(NewUserMessage):
    def __init__(
        self,
        content: str | list[UserMessageContent] | None = None,
        *,
        role: Literal["user"] = "user",
        meta: MessageMeta | None = None,
    ):
        if isinstance(content, str):
            content = [UserTextContent(text=content)]
        elif content is None:
            content = []
        super().__init__(
            role=role,
            content=content,
            meta=meta or MessageMeta(),
        )


class AgentAssistantMessage(NewAssistantMessage):
    def __init__(
        self,
        content: str | list[AssistantMessageContent] | None = None,
        *,
        role: Literal["assistant"] = "assistant",
        meta: AssistantMessageMeta | None = None,
    ):
        if isinstance(content, str):
            content = [AssistantTextContent(text=content)]
        elif content is None:
            content = []
        super().__init__(
            role=role,
            content=content,
            meta=meta or AssistantMessageMeta(),
        )


AgentSystemMessage = NewSystemMessage
RunnerMessage = NewMessage


# Streaming processor types
class AssistantMessage(BaseModel):
    """
    Temporary assistant message used during streaming processing.

    This is a simplified message format used internally by completion event processors
    to accumulate streaming content before converting to the final NewAssistantMessage format.
    """

    role: Literal["assistant"] = "assistant"
    id: str = ""
    index: int | None = None
    content: str = ""
    tool_calls: list[Any] | None = None


# Enhanced type definitions for better type hints
# Supports new message format, legacy messages, and dict (for backward compatibility)
FlexibleRunnerMessage = NewMessage | AgentUserMessage | AgentAssistantMessage | dict[str, Any]
RunnerMessages = Sequence[FlexibleRunnerMessage]


# Type alias for user input - supports string, single message, or sequence of messages
UserInput = str | FlexibleRunnerMessage | RunnerMessages


def user_message_to_llm_dict(message: NewUserMessage) -> dict[str, Any]:
    """Convert NewUserMessage to dict for LLM API"""
    # Convert content to simplified format for LLM
    content = message.content[0].text if len(message.content) == 1 and message.content[0].type == "text" else [item.model_dump() for item in message.content]
    return {"role": message.role, "content": content}


def system_message_to_llm_dict(message: NewSystemMessage) -> dict[str, Any]:
    """Convert NewSystemMessage to dict for LLM API"""
    return {"role": message.role, "content": message.content}


def assistant_message_to_llm_dict(message: NewAssistantMessage) -> dict[str, Any]:
    """Convert NewAssistantMessage to dict for LLM API"""
    # Separate text content from tool calls
    text_parts = []
    tool_calls = []

    for item in message.content:
        if item.type == "text":
            text_parts.append(item.text)
        elif item.type == "tool_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments if isinstance(item.arguments, str) else str(item.arguments),
                    },
                },
            )

    result = {
        "role": message.role,
        "content": " ".join(text_parts) if text_parts else None,
    }

    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


def message_to_llm_dict(message: NewMessage) -> dict[str, Any]:
    """Convert any NewMessage to dict for LLM API"""
    if isinstance(message, NewUserMessage):
        return user_message_to_llm_dict(message)
    if isinstance(message, NewSystemMessage):
        return system_message_to_llm_dict(message)
    if isinstance(message, NewAssistantMessage):
        return assistant_message_to_llm_dict(message)
    # Fallback
    return message.model_dump(exclude={"meta"})


def messages_to_llm_format(messages: Sequence[NewMessage]) -> list[dict[str, Any]]:
    """Convert a sequence of NewMessage to LLM format, excluding meta data"""
    return [message_to_llm_dict(message) for message in messages]
