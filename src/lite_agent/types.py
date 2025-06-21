from typing import Literal, TypedDict

from litellm import Usage
from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: str | None = None


class ToolCall(BaseModel):
    type: Literal["function"]
    function: ToolCallFunction
    id: str


class AssistantMessage(BaseModel):
    id: str
    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] | None = None


class Message(TypedDict):
    role: str
    content: str


class UserMessageContentItemText(TypedDict):
    type: Literal["text"]
    text: str


class UserMessageContentItemImageURLImageURL(TypedDict):
    url: str


class UserMessageContentItemImageURL(TypedDict):
    type: Literal["image_url"]
    image_url: UserMessageContentItemImageURLImageURL


class AgentUserMessage(TypedDict):
    role: Literal["user"]
    content: str | list[UserMessageContentItemText | UserMessageContentItemImageURL]


class AssistantMessageToolCallFunction(TypedDict):
    name: str
    arguments: str


class AssistantMessageToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: AssistantMessageToolCallFunction
    tool_call_id: str


class AgentAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str
    tool_calls: list[AssistantMessageToolCall] | None


class AgentSystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class AgentToolCallMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


RunnerMessage = AgentUserMessage | AgentAssistantMessage | AgentToolCallMessage
AgentMessage = RunnerMessage | AgentSystemMessage
RunnerMessages = list[RunnerMessage]


class LiteLLMRawChunk(TypedDict):
    """
    Define the type of chunk
    """

    type: Literal["litellm_raw"]
    raw: ModelResponseStream


class UsageChunk(TypedDict):
    """
    Define the type of usage info chunk
    """

    type: Literal["usage"]
    usage: Usage


class FinalMessageChunk(TypedDict):
    """
    Define the type of final message chunk
    """

    type: Literal["final_message"]
    message: AssistantMessage
    finish_reason: Literal["stop", "tool_calls"]


class ToolCallChunk(TypedDict):
    """
    Define the type of tool call chunk
    """

    type: Literal["tool_call"]
    name: str
    arguments: str


class ToolCallResultChunk(TypedDict):
    """
    Define the type of tool call result chunk
    """

    type: Literal["tool_call_result"]
    tool_call_id: str
    name: str
    content: str


class ContentDeltaChunk(TypedDict):
    """
    Define the type of message chunk
    """

    type: Literal["content_delta"]
    delta: str


class ToolCallDeltaChunk(TypedDict):
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
