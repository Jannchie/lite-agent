from typing import Any

from funcall import Funcall
from rich import print  # noqa: A004

from lite_agent.client import LiteLLMClient
from lite_agent.stream_handlers import litellm_response_stream_handler


def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    return f"The temperature in {city} is 25°C."


def get_whether(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny with a few clouds."


fc = Funcall([get_temperature, get_whether])

messages: list[dict[str, Any]] = [
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


async def main():
    client = LiteLLMClient(model="gpt-4.1-nano")
    resp = await client.responses(messages=messages, tools=fc.get_tools(), tool_choice="auto")
    handler_resp = litellm_response_stream_handler(resp)
    async for event in handler_resp:
        print(event)

    # Close any remaining aiohttp sessions


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
