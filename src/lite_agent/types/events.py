from typing import Literal

from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel

from .messages import AgentAssistantMessage


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class CompletionRawEvent(BaseModel):
    """
    Define the type of chunk
    """

    type: Literal["completion_raw"] = "completion_raw"
    raw: ModelResponseStream


class UsageEvent(BaseModel):
    """
    Define the type of usage info chunk
    """

    type: Literal["usage"] = "usage"
    usage: Usage


class AssistantMessageEvent(BaseModel):
    """
    Define the type of assistant message chunk
    """

    type: Literal["assistant_message"] = "assistant_message"
    message: AgentAssistantMessage


class FunctionCallEvent(BaseModel):
    """
    Define the type of tool call chunk
    """

    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


class FunctionCallOutputEvent(BaseModel):
    """
    Define the type of tool call result chunk
    """

    type: Literal["function_call_output"] = "function_call_output"
    tool_call_id: str
    name: str
    content: str


class ContentDeltaEvent(BaseModel):
    """
    Define the type of message chunk
    """

    type: Literal["content_delta"] = "content_delta"
    delta: str


class FunctionCallDeltaEvent(BaseModel):
    """
    Define the type of tool call delta chunk
    """

    type: Literal["function_call_delta"] = "function_call_delta"
    tool_call_id: str
    name: str
    arguments_delta: str


AgentChunk = CompletionRawEvent | UsageEvent | FunctionCallEvent | FunctionCallOutputEvent | ContentDeltaEvent | FunctionCallDeltaEvent | AssistantMessageEvent

AgentChunkType = Literal[
    "completion_raw",
    "usage",
    "function_call",
    "function_call_output",
    "content_delta",
    "function_call_delta",
    "assistant_message",
]
