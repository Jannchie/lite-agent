from typing import Any

from funcall import Funcall
from litellm.types.llms.openai import (
    ContentPartAddedEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from openai.types.responses.response_input_param import ResponseInputParam
from rich import print  # noqa: A004

from lite_agent.client import LiteLLMClient
from lite_agent.types import AgentAssistantMessage, AssistantMessageEvent, ContentDeltaEvent, FunctionCallEvent, UsageEvent
from lite_agent.types.events import Usage


def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    return f"The temperature in {city} is 25°C."


def get_whether(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny with a few clouds."


fc = Funcall([get_temperature, get_whether])

messages: ResponseInputParam = [
    {
        "role": "system",
        "content": "You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
    },
    {
        "role": "user",
        "content": "What is the weather in New York?",
    },
    # {
    #     "arguments": '{"city":"New York"}',
    #     "call_id": "call_V23uiXgXRlv9pRoW4qAgflKF",
    #     "name": "get_whether",
    #     "type": "function_call",
    # },
    # {
    #     "call_id": "call_V23uiXgXRlv9pRoW4qAgflKF",
    #     "output": '{"result": 42, "msg": "done"}',
    #     "type": "function_call_output",
    # },
]


def handle_message(messages: list[dict[str, Any]], event: ResponsesAPIStreamingResponse):  # noqa: C901, PLR0912
    if event.type in (
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE,
    ):
        ...
    elif isinstance(event, OutputItemAddedEvent):
        messages.append(event.item)  # type: ignore
    elif isinstance(event, ContentPartAddedEvent):
        latest_message = messages[-1] if messages else None
        if latest_message and isinstance(latest_message["content"], list):
            latest_message["content"].append(event.part)
    elif isinstance(event, OutputTextDeltaEvent):
        latest_message = messages[-1] if messages else None
        if latest_message and isinstance(latest_message["content"], list):
            latest_content = latest_message["content"][-1]
            latest_content["text"] += event.delta
            return ContentDeltaEvent(delta=event.delta)
    elif isinstance(event, OutputItemDoneEvent):
        item = event.item
        if item.get("type") == "function_call":
            return FunctionCallEvent(
                call_id=item["call_id"],
                name=item["name"],
                arguments=item["arguments"],
            )
        if item.get("type") == "message":
            return AssistantMessageEvent(
                message=AgentAssistantMessage(content=item["content"][0]["text"]),
            )
    elif isinstance(event, FunctionCallArgumentsDeltaEvent):
        if messages:
            latest_message = messages[-1]
            if latest_message["type"] == "function_call":
                if "arguments" not in latest_message:
                    latest_message["arguments"] = ""
                latest_message["arguments"] += event.delta
    elif isinstance(event, FunctionCallArgumentsDoneEvent):
        if messages:
            latest_message = messages[-1]
            if latest_message["type"] == "function_call":
                latest_message["arguments"] = event.arguments
    elif isinstance(event, ResponseCompletedEvent):
        usage = event.response.usage
        if usage:
            return UsageEvent(
                usage=Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
            )
    return None


async def main():
    client = LiteLLMClient(model="gpt-4.1-nano")
    resp = await client.responses(messages=messages, tools=fc.get_tools(), tool_choice="auto")
    new_messages: list[dict[str, str]] = []
    async for event in resp:  # type: ignore
        # print(event)
        e = handle_message(new_messages, event)
        if e:
            print(e)
    print(new_messages)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
