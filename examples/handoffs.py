#!/usr/bin/env python3
"""Example showing how to use handoffs between agents.

This example demonstrates how agents can transfer conversations to each other
using automatically generated transfer functions.
"""

import asyncio

from lite_agent.agent import Agent
from lite_agent.runner import Runner


async def main():
    """Demonstrate agent handoffs functionality."""
    child = Agent(
        model="gpt-4.1",
        name="ChildAgent",
        instructions="You are a helpful assistant.",
    )

    parent = Agent(
        model="gpt-4.1",
        name="ParentAgent",
        instructions="You will transfer conversations to ChildAgent.",
        handoffs=[child],
    )
    runner = Runner(parent)
    resp = runner.run("Hello, I need help with my order.", includes=["final_message", "tool_call", "tool_call_result"])
    async for message in resp:
        print(message)
    print(runner.agent.name)


if __name__ == "__main__":
    asyncio.run(main())
