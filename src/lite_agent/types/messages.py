from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel
from rich import Any

from .tool_calls import ToolCall


# Response API format input types
class ResponseInputText(BaseModel):
    text: str
    type: Literal["input_text"]


class ResponseInputImage(BaseModel):
    detail: Literal["low", "high", "auto"] = "auto"
    type: Literal["input_image"]
    file_id: str | None = None
    image_url: str | None = None


class ResponseInputTextParam(TypedDict):
    text: str
    type: Literal["input_text"]


class ResponseInputImageParam(TypedDict):
    detail: NotRequired[Literal["low", "high", "auto"]]
    type: Literal["input_image"]
    file_id: str | None
    image_url: str | None


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
    id: str
    index: int
    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] | None = None


class Message(BaseModel):
    role: str
    content: str


class AgentUserMessage(BaseModel):
    role: Literal["user"]
    content: str | Sequence[ResponseInputText | ResponseInputImage | UserMessageContentItemText | UserMessageContentItemImageURL]


class AgentAssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class AgentSystemMessage(BaseModel):
    role: Literal["system"]
    content: str


class AgentFunctionToolCallMessage(BaseModel):
    arguments: str
    type: Literal["function_call"]
    function_call_id: str
    name: str
    content: str


class AgentFunctionCallOutput(BaseModel):
    call_id: str
    output: str
    type: Literal["function_call_output"]


RunnerMessage = AgentUserMessage | AgentAssistantMessage | AgentSystemMessage | AgentFunctionToolCallMessage | AgentFunctionCallOutput
AgentMessage = RunnerMessage | AgentSystemMessage
RunnerMessages = Sequence[RunnerMessage | dict[str, Any]]
