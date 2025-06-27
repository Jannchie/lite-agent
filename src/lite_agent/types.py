from collections.abc import Sequence
from typing import Literal

from litellm import Usage
from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel
from rich import Any


class ToolCallFunction(BaseModel):
    name: str
    arguments: str | None = None


class ToolCall(BaseModel):
    type: Literal["function"]
    function: ToolCallFunction
    id: str
    index: int


class AssistantMessage(BaseModel):
    id: str
    index: int
    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: Sequence[ToolCall] | None = None


class Message(BaseModel):
    role: str
    content: str


class UserMessageContentItemText(BaseModel):
    type: Literal["text"]
    text: str


class UserMessageContentItemImageURLImageURL(BaseModel):
    url: str


class UserMessageContentItemImageURL(BaseModel):
    type: Literal["image_url"]
    image_url: UserMessageContentItemImageURLImageURL


class AgentUserMessage(BaseModel):
    role: Literal["user"]
    content: str | Sequence[UserMessageContentItemText | UserMessageContentItemImageURL]


class AgentAssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str
    tool_calls: Sequence[ToolCall] | None = None


class AgentSystemMessage(BaseModel):
    role: Literal["system"]
    content: str


class AgentToolCallMessage(BaseModel):
    role: Literal["tool"]
    tool_call_id: str
    content: str


RunnerMessage = AgentUserMessage | AgentAssistantMessage | AgentToolCallMessage | AgentSystemMessage
AgentMessage = RunnerMessage | AgentSystemMessage
RunnerMessages = Sequence[RunnerMessage | dict[str, Any]]


class LiteLLMRawChunk(BaseModel):
    """
    Define the type of chunk
    """

    type: Literal["litellm_raw"]
    raw: ModelResponseStream


class UsageChunk(BaseModel):
    """
    Define the type of usage info chunk
    """

    type: Literal["usage"]
    usage: Usage


class FinalMessageChunk(BaseModel):
    """
    Define the type of final message chunk
    """

    type: Literal["final_message"]
    message: AssistantMessage
    finish_reason: str | None = None  # Literal["stop", "tool_calls"]


class ToolCallChunk(BaseModel):
    """
    Define the type of tool call chunk
    """

    type: Literal["tool_call"]
    name: str
    arguments: str


class ToolCallResultChunk(BaseModel):
    """
    Define the type of tool call result chunk
    """

    type: Literal["tool_call_result"]
    tool_call_id: str
    name: str
    content: str


class ContentDeltaChunk(BaseModel):
    """
    Define the type of message chunk
    """

    type: Literal["content_delta"]
    delta: str


class ToolCallDeltaChunk(BaseModel):
    """
    Define the type of tool call delta chunk
    """

    type: Literal["tool_call_delta"]
    tool_call_id: str
    name: str
    arguments_delta: str


AgentChunk = LiteLLMRawChunk | UsageChunk | FinalMessageChunk | ToolCallChunk | ToolCallResultChunk | ContentDeltaChunk | ToolCallDeltaChunk

AgentChunkType = Literal[
    "litellm_raw",
    "usage",
    "final_message",
    "tool_call",
    "tool_call_result",
    "content_delta",
    "tool_call_delta",
]
