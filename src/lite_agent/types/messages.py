from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field, model_validator

from .tool_calls import ToolCall


class BasicMessageMeta(BaseModel):
    """Basic metadata for user messages and function calls"""

    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: int | None = None


class LLMResponseMeta(BaseModel):
    """Metadata for LLM responses, includes performance metrics"""

    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: int | None = None
    output_time_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


# New unified metadata types
class MessageMeta(BaseModel):
    """Base metadata for all message types"""

    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM API"""
        # Convert content to simplified format for LLM
        content = self.content[0].text if len(self.content) == 1 and self.content[0].type == "text" else [item.model_dump() for item in self.content]
        return {"role": self.role, "content": content}


class NewSystemMessage(BaseModel):
    """System message"""

    role: Literal["system"] = "system"
    content: str
    meta: MessageMeta = Field(default_factory=MessageMeta)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM API"""
        return {"role": self.role, "content": self.content}


class NewAssistantMessage(BaseModel):
    """Assistant message with structured content and metadata"""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantMessageContent]
    meta: AssistantMessageMeta = Field(default_factory=AssistantMessageMeta)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM API"""
        # Separate text content from tool calls
        text_parts = []
        tool_calls = []

        for item in self.content:
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
            "role": self.role,
            "content": " ".join(text_parts) if text_parts else None,
        }

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result


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


# Legacy types - keeping for compatibility
class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    id: str
    index: int | None = None
    content: str = ""
    tool_calls: list[ToolCall] | None = None


class Message(BaseModel):
    role: str
    content: str


class AgentUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | Sequence[ResponseInputText | ResponseInputImage | UserMessageContentItemText | UserMessageContentItemImageURL]
    meta: BasicMessageMeta = Field(default_factory=BasicMessageMeta)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM, excluding meta data"""
        return {"role": self.role, "content": self.content}


class AgentAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    meta: LLMResponseMeta = Field(default_factory=LLMResponseMeta)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM, excluding meta data"""
        return {"role": self.role, "content": self.content}


class AgentSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str
    meta: BasicMessageMeta = Field(default_factory=BasicMessageMeta)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM, excluding meta data"""
        return {"role": self.role, "content": self.content}


class AgentFunctionToolCallMessage(BaseModel):
    type: Literal["function_call"] = "function_call"
    arguments: str
    call_id: str
    name: str
    meta: BasicMessageMeta = Field(default_factory=BasicMessageMeta)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM, excluding meta data"""
        return {
            "type": self.type,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "name": self.name,
        }


class AgentFunctionCallOutput(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str
    meta: BasicMessageMeta = Field(default_factory=BasicMessageMeta)

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM, excluding meta data"""
        return {
            "type": self.type,
            "call_id": self.call_id,
            "output": self.output,
        }


RunnerMessage = AgentUserMessage | AgentAssistantMessage | AgentSystemMessage | AgentFunctionToolCallMessage | AgentFunctionCallOutput
AgentMessage = RunnerMessage | AgentSystemMessage

# Enhanced type definitions for better type hints
# Supports BaseModel instances, TypedDict, and plain dict
FlexibleRunnerMessage = RunnerMessage | NewMessage | MessageDict | dict[str, Any]
RunnerMessages = Sequence[FlexibleRunnerMessage]

# Type alias for user input - supports string, single message, or sequence of messages
UserInput = str | FlexibleRunnerMessage | RunnerMessages


def messages_to_llm_format(messages: Sequence[RunnerMessage]) -> list[dict[str, Any]]:
    """Convert a sequence of RunnerMessage to LLM format, excluding meta data"""
    result = []
    for message in messages:
        if hasattr(message, "to_llm_dict"):
            result.append(message.to_llm_dict())
        else:
            # Fallback for messages without to_llm_dict method
            result.append(message.model_dump(exclude={"meta"}))
    return result


# Conversion functions between old and new message formats
def convert_legacy_to_new(messages: Sequence[RunnerMessage]) -> list[NewMessage]:
    """Convert legacy message format to new structured format"""
    result: list[NewMessage] = []
    i = 0

    while i < len(messages):
        message = messages[i]

        if isinstance(message, AgentUserMessage):
            # Convert user message
            if isinstance(message.content, str):
                user_content: list[UserMessageContent] = [UserTextContent(text=message.content)]
            else:
                user_content: list[UserMessageContent] = []
                for item in message.content:
                    if hasattr(item, "text"):
                        user_content.append(UserTextContent(text=getattr(item, "text", "")))
                    elif hasattr(item, "image_url"):
                        image_url_attr = getattr(item, "image_url", None)
                        if image_url_attr is not None:
                            image_url = getattr(image_url_attr, "url", str(image_url_attr)) if hasattr(image_url_attr, "url") else str(image_url_attr)
                            user_content.append(UserImageContent(image_url=image_url))
                    # Add more conversion logic as needed

            result.append(
                NewUserMessage(
                    content=user_content,
                    meta=MessageMeta(sent_at=message.meta.sent_at),
                ),
            )

        elif isinstance(message, AgentSystemMessage):
            result.append(
                NewSystemMessage(
                    content=message.content,
                    meta=MessageMeta(sent_at=message.meta.sent_at),
                ),
            )

        elif isinstance(message, AgentAssistantMessage):
            # Look ahead for related tool calls and results
            assistant_content: list[AssistantMessageContent] = []

            # Add text content
            if message.content:
                assistant_content.append(AssistantTextContent(text=message.content))

            # Collect tool calls and results that follow
            j = i + 1
            while j < len(messages):
                next_message = messages[j]
                if isinstance(next_message, AgentFunctionToolCallMessage):
                    assistant_content.append(
                        AssistantToolCall(
                            call_id=next_message.call_id,
                            name=next_message.name,
                            arguments=next_message.arguments,
                        ),
                    )
                    j += 1
                elif isinstance(next_message, AgentFunctionCallOutput):
                    # Find matching tool call
                    assistant_content.append(
                        AssistantToolCallResult(
                            call_id=next_message.call_id,
                            output=next_message.output,
                            execution_time_ms=next_message.meta.execution_time_ms,
                        ),
                    )
                    j += 1
                else:
                    break

            # Create assistant message meta with enhanced data
            assistant_meta = AssistantMessageMeta(sent_at=message.meta.sent_at)
            if hasattr(message.meta, "latency_ms"):
                assistant_meta.latency_ms = message.meta.latency_ms
            if hasattr(message.meta, "input_tokens") and hasattr(message.meta, "output_tokens"):
                assistant_meta.usage = MessageUsage(
                    input_tokens=message.meta.input_tokens,
                    output_tokens=message.meta.output_tokens,
                    total_tokens=(message.meta.input_tokens or 0) + (message.meta.output_tokens or 0),
                )
            if hasattr(message.meta, "output_time_ms"):
                assistant_meta.total_time_ms = message.meta.output_time_ms

            result.append(
                NewAssistantMessage(
                    content=assistant_content,
                    meta=assistant_meta,
                ),
            )

            # Skip the processed tool calls and results
            i = j - 1

        i += 1

    return result


def convert_new_to_legacy(messages: Sequence[NewMessage]) -> list[RunnerMessage]:
    """Convert new structured format to legacy message format"""
    result: list[RunnerMessage] = []

    for message in messages:
        if isinstance(message, NewUserMessage):
            # Convert to legacy user message
            if len(message.content) == 1 and message.content[0].type == "text":
                content = message.content[0].text
            else:
                # Convert to legacy multi-content format
                content = []
                for item in message.content:
                    if item.type == "text":
                        content.append(ResponseInputText(text=item.text))
                    elif item.type == "image":
                        content.append(
                            ResponseInputImage(
                                image_url=item.image_url,
                                file_id=item.file_id,
                                detail=item.detail,
                            ),
                        )

            result.append(
                AgentUserMessage(
                    content=content,
                    meta=BasicMessageMeta(sent_at=message.meta.sent_at),
                ),
            )

        elif isinstance(message, NewSystemMessage):
            result.append(
                AgentSystemMessage(
                    content=message.content,
                    meta=BasicMessageMeta(sent_at=message.meta.sent_at),
                ),
            )

        elif isinstance(message, NewAssistantMessage):
            # Extract text content first
            text_parts = [item.text for item in message.content if item.type == "text"]

            # Create assistant message
            assistant_meta = LLMResponseMeta(sent_at=message.meta.sent_at)
            if message.meta.latency_ms:
                assistant_meta.latency_ms = message.meta.latency_ms
            if message.meta.total_time_ms:
                assistant_meta.output_time_ms = message.meta.total_time_ms
            if message.meta.usage:
                assistant_meta.input_tokens = message.meta.usage.input_tokens
                assistant_meta.output_tokens = message.meta.usage.output_tokens

            result.append(
                AgentAssistantMessage(
                    content=" ".join(text_parts),
                    meta=assistant_meta,
                ),
            )

            # Add tool calls and results as separate messages
            for item in message.content:
                if item.type == "tool_call":
                    result.append(
                        AgentFunctionToolCallMessage(
                            call_id=item.call_id,
                            name=item.name,
                            arguments=item.arguments if isinstance(item.arguments, str) else str(item.arguments),
                            meta=BasicMessageMeta(sent_at=message.meta.sent_at),
                        ),
                    )
                elif item.type == "tool_call_result":
                    result.append(
                        AgentFunctionCallOutput(
                            call_id=item.call_id,
                            output=item.output,
                            meta=BasicMessageMeta(
                                sent_at=message.meta.sent_at,
                                execution_time_ms=item.execution_time_ms,
                            ),
                        ),
                    )

    return result
