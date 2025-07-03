from collections.abc import Sequence
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel

from .tool_calls import ToolCall


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
    function_call_id: str
    name: str
    arguments: str
    content: str


class FunctionCallOutputDict(TypedDict):
    type: Literal["function_call_output"]
    call_id: str
    output: str


# Union type for all supported message dictionary formats
MessageDict = UserMessageDict | AssistantMessageDict | SystemMessageDict | FunctionCallDict | FunctionCallOutputDict


# Response API format input types
class ResponseInputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class ResponseInputImage(BaseModel):
    detail: Literal["low", "high", "auto"] = "auto"
    type: Literal["input_image"] = "input_image"
    file_id: str | None = None
    image_url: str | None = None


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


class AgentAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class AgentSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class AgentFunctionToolCallMessage(BaseModel):
    type: Literal["function_call"] = "function_call"
    arguments: str
    function_call_id: str
    name: str


class AgentFunctionCallOutput(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


RunnerMessage = AgentUserMessage | AgentAssistantMessage | AgentSystemMessage | AgentFunctionToolCallMessage | AgentFunctionCallOutput
AgentMessage = RunnerMessage | AgentSystemMessage

# Enhanced type definitions for better type hints
# Supports BaseModel instances, TypedDict, and plain dict
FlexibleRunnerMessage = RunnerMessage | MessageDict | dict[str, Any]
RunnerMessages = Sequence[FlexibleRunnerMessage]

# Type alias for user input - supports string, single message, or sequence of messages
UserInput = str | FlexibleRunnerMessage | RunnerMessages
