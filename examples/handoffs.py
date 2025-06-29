#!/usr/bin/env python3
"""Example showing how to use handoffs between agents.

This example demonstrates how agents can transfer conversations to each other
using automatically generated transfer functions.
"""

import asyncio
import logging

from rich.logging import RichHandler

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
    # In a real application, this would call an API or database.
    return f"The temperature in {city} is 25°C."


def get_weather(city: str) -> str:
    """Get the weather of a city."""
    # In a real application, this would call an API or database.
    return f"The weather in {city} is sunny."


async def main():
    """Demonstrate agent handoffs functionality."""
    parent = Agent(
        model="gpt-4.1",
        name="ParentAgent",
        instructions="You are a helpful agent. You can transfer conversations to other agents for specific tasks. You cannot transfer to multiple agents at once. You should transfer step by step.",
    )

    whether_agent = Agent(
        model="gpt-4.1",
        name="WhetherAgent",
        instructions="You are a helpful agent to check weather",
        tools=[get_weather],
    )

    temper_agent = Agent(
        model="gpt-4.1",
        name="TemperatureAgent",
        instructions="You are a helpful agent to check temperature",
        tools=[get_temperature],
    )

    parent.add_handoff(whether_agent)
    parent.add_handoff(temper_agent)

    runner = Runner(parent)
    resp = runner.run("Hello, I need to check the whether and temperature of Tokyo.", includes=["final_message", "tool_call", "tool_call_result"])
    async for message in resp:
        logger.info(message)
    logger.info(runner.agent.name)


if __name__ == "__main__":
    asyncio.run(main())
