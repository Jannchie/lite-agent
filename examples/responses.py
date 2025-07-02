from typing import Any

import litellm
from funcall import Funcall
from litellm.types.llms.openai import (
    ContentPartAddedEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputTextDeltaEvent,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from openai.types.responses.response_input_param import ResponseInputParam


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

resp = litellm.responses(model="gpt-4.1-nano", input=messages, tools=fc.get_tools(), tool_choice="auto", stream=True, store=False)


def handle_message(messages: list[dict[str, Any]], event: ResponsesAPIStreamingResponse):
    if event.type == ResponsesAPIStreamEvents.RESPONSE_CREATED:  # noqa: SIM114
        pass
    elif event.type == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS:
        pass
    elif isinstance(event, OutputItemAddedEvent):
        messages.append(event.item)  # type: ignore
    elif isinstance(event, ContentPartAddedEvent):
        latest_message = messages[-1] if messages else None
        if latest_message and isinstance(latest_message["content"], list):
            latest_message["content"].append(event.part)  # type: ignore
    elif isinstance(event, OutputTextDeltaEvent):
        latest_message = messages[-1] if messages else None
        if latest_message and isinstance(latest_message["content"], list):
            latest_content = latest_message["content"][-1]
            latest_content["text"] += event.delta  # type: ignore
    elif event.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:  # noqa: SIM114
        pass
    elif event.type == ResponsesAPIStreamEvents.CONTENT_PART_DONE:  # noqa: SIM114
        pass
    elif event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
        pass
    # OUTPUT_ITEM_ADDED
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
    # OUTPUT_ITEM_ADDED
    elif event.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
        pass


new_messages: list[dict[str, str]] = []
for event in resp:  # type: ignore
    # print(message.model_dump())
    handle_message(new_messages, event)
print(new_messages)
