from typing import Literal

from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel

from .messages import NewAssistantMessage


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class Timing(BaseModel):
    latency_ms: int
    output_time_ms: int


class CompletionRawEvent(BaseModel):
    """
    Define the type of chunk
    """

    type: Literal["completion_raw"] = "completion_raw"
    raw: ModelResponseStream


class ResponseRawEvent(BaseModel):
    """
    Define the type of response raw chunk
    """

    type: Literal["response_raw"] = "response_raw"
    raw: object


class UsageEvent(BaseModel):
    """
    Define the type of usage info chunk
    """

    type: Literal["usage"] = "usage"
    usage: Usage


class TimingEvent(BaseModel):
    """
    Define the type of timing info chunk
    """

    type: Literal["timing"] = "timing"
    timing: Timing


class AssistantMessageEvent(BaseModel):
    """
    Define the type of assistant message chunk
    """

    type: Literal["assistant_message"] = "assistant_message"
    message: NewAssistantMessage


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
    execution_time_ms: int | None = None


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


AgentChunk = CompletionRawEvent | ResponseRawEvent | UsageEvent | TimingEvent | FunctionCallEvent | FunctionCallOutputEvent | ContentDeltaEvent | FunctionCallDeltaEvent | AssistantMessageEvent

AgentChunkType = Literal[
    "completion_raw",
    "response_raw",
    "usage",
    "timing",
    "function_call",
    "function_call_output",
    "content_delta",
    "function_call_delta",
    "assistant_message",
]
