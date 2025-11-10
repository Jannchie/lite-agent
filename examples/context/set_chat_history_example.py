#!/usr/bin/env python3
"""Example demonstrating the set_chat_history functionality.

This example shows how to set a complete chat history and automatically track
which agent should be active based on function calls in the history.
"""

import asyncio
import logging

from rich.logging import RichHandler

from lite_agent import consolidate_history_transfer
from lite_agent.agent import Agent
from lite_agent.loggers import logger
from lite_agent.runner import Runner

logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger.setLevel(logging.DEBUG)


def get_temperature(city: str) -> str:
    """Get the temperature of a city."""
    return f"The temperature in {city} is 25°C."


def get_weather(city: str) -> str:
    """Get the weather of a city."""
    return f"The weather in {city} is sunny."


async def main():
    """Demonstrate set_chat_history functionality."""
    # Create agents
    parent = Agent(
        model="gpt-4.1",
        name="ParentAgent",
        instructions="You are a helpful agent.",
    )

    weather_agent = Agent(
        model="gpt-4.1",
        name="WeatherAgent",
        instructions="You are a helpful agent to check weather.",
        tools=[get_weather],
        message_transfer=consolidate_history_transfer,
    )

    temperature_agent = Agent(
        model="gpt-4.1",
        name="TemperatureAgent",
        instructions="You are a helpful agent to check temperature.",
        tools=[get_temperature],
        message_transfer=consolidate_history_transfer,
    )

    parent.add_handoff(weather_agent)
    parent.add_handoff(temperature_agent)

    # Create runner
    runner = Runner(parent)

    # Example chat history that includes agent transfers
    chat_history = [
        {"role": "user", "content": "Hello, I need to check the whether and temperature of Tokyo."},
        {"role": "assistant", "content": ""},
        {"arguments": '{"name":"WhetherAgent"}', "type": "function_call", "call_id": "call_HRTCM7KqkMkwhFWs2THxeub1", "name": "transfer_to_agent", "content": ""},
        {"call_id": "call_HRTCM7KqkMkwhFWs2THxeub1", "output": "Transferring to agent: WhetherAgent", "type": "function_call_output"},
        {"role": "assistant", "content": ""},
        {"arguments": '{"city":"Tokyo"}', "type": "function_call", "call_id": "call_W02qQluI5XeM6Pk7je8xEwVs", "name": "get_weather", "content": ""},
        {"call_id": "call_W02qQluI5XeM6Pk7je8xEwVs", "output": "The weather in Tokyo is sunny.", "type": "function_call_output"},
        {"role": "assistant", "content": ""},
        {"arguments": "{}", "type": "function_call", "call_id": "call_wnnlXwlusLSLygIppk4l5TkD", "name": "transfer_to_parent", "content": ""},
        {"call_id": "call_wnnlXwlusLSLygIppk4l5TkD", "output": "Transferring back to parent agent: ParentAgent", "type": "function_call_output"},
        {"role": "assistant", "content": ""},
        {"arguments": '{"name":"TemperatureAgent"}', "type": "function_call", "call_id": "call_S9OlXSk7kEPQJ3HWC3GoK0xE", "name": "transfer_to_agent", "content": ""},
        {"call_id": "call_S9OlXSk7kEPQJ3HWC3GoK0xE", "output": "Transferring to agent: TemperatureAgent", "type": "function_call_output"},
        {"role": "assistant", "content": ""},
        {"arguments": '{"city":"Tokyo"}', "type": "function_call", "call_id": "call_gDljMYP187ymVXfKCUdsgjlX", "name": "get_temperature", "content": ""},
        {"call_id": "call_gDljMYP187ymVXfKCUdsgjlX", "output": "The temperature in Tokyo is 25°C.", "type": "function_call_output"},
        {"role": "assistant", "content": ""},
        {"arguments": "{}", "type": "function_call", "call_id": "call_bQiSAr8y7u6Q0rJMqUfVE47p", "name": "transfer_to_parent", "content": ""},
        {"call_id": "call_bQiSAr8y7u6Q0rJMqUfVE47p", "output": "Transferring back to parent agent: ParentAgent", "type": "function_call_output"},
        {"role": "assistant", "content": "The weather in Tokyo is sunny, and the temperature is 25°C. If you need further details about Tokyo or want updates for another city, let me know!"},
    ]
    # Set the chat history - this should automatically track that we should be on TemperatureAgent
    runner.set_chat_history(chat_history)

    print(f"After setting chat history, current agent is: {runner.agent.name}")
    print(f"Chat history contains {len(runner.messages)} messages")

    # Continue the conversation from where we left off
    print("\n--- Continuing conversation ---")
    resp = runner.run([], max_steps=5, includes=["assistant_message", "function_call", "function_call_output"])
    async for message in resp:
        logger.info(message)


if __name__ == "__main__":
    asyncio.run(main())
