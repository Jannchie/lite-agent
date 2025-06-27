from typing import Literal

from litellm import Usage
from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel

from .messages import AssistantMessage


class CompletionRawChunk(BaseModel):
    """
    Define the type of chunk
    """

    type: Literal["completion_raw"]
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


AgentChunk = CompletionRawChunk | UsageChunk | FinalMessageChunk | ToolCallChunk | ToolCallResultChunk | ContentDeltaChunk | ToolCallDeltaChunk

AgentChunkType = Literal[
    "completion_raw",
    "usage",
    "final_message",
    "tool_call",
    "tool_call_result",
    "content_delta",
    "tool_call_delta",
]
