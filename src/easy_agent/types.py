from typing import Literal, TypedDict


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
    role: Literal["user"] = "user"
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
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: list[AssistantMessageToolCall] | None


class AgentSystemMessage(TypedDict):
    role: Literal["system"] = "system"
    content: str


class AgentToolCallMessage(TypedDict):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


RunnerMessage = AgentUserMessage | AgentAssistantMessage | AgentToolCallMessage
AgentMessage = RunnerMessage | AgentSystemMessage
RunnerMessages = list[RunnerMessage]
