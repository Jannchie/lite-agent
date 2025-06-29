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


async def main():
    """Demonstrate agent handoffs functionality."""
    child = Agent(
        model="gpt-4.1",
        name="ChildAgent",
        instructions="You are a helpful assistant.",
    )

    # Create another child agent
    child2 = Agent(
        model="gpt-4.1",
        name="ChildAgent2",
        instructions="You are a specialized assistant for advanced tasks.",
    )

    parent = Agent(
        model="gpt-4.1",
        name="ParentAgent",
        instructions="You will transfer conversations to ChildAgent.",
        handoffs=[child],
    )

    # Demonstrate adding handoff after initialization
    parent.add_handoff(child2)

    runner = Runner(parent)
    resp = runner.run("Hello, I need help with my order.", includes=["final_message", "tool_call", "tool_call_result"])
    async for message in resp:
        logger.info(message)
    logger.info(runner.agent.name)


if __name__ == "__main__":
    asyncio.run(main())
