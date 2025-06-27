from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel
from rich import Any

from .tool_calls import ToolCall


class AssistantMessage(BaseModel):
    id: str
    index: int
    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] | None = None


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
