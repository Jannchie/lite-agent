"""Test script for completion_condition feature."""

import asyncio
import os

from lite_agent import Agent, Runner


def simple_greeting(name: str) -> str:
    """Simple greeting function."""
    return f"Hello, {name}!"


async def test_stop_condition():
    """Test default stop condition."""
    print("Testing stop condition...")

    agent = Agent(
        name="StopAgent",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
        tools=[simple_greeting],
        completion_condition="stop",
    )

    runner = Runner(agent)
    await runner.run_until_complete("Say hello to Alice.")

    print(f"Number of messages: {len(runner.messages)}")
    print("Test completed!")


async def test_call_condition():
    """Test call completion condition."""
    print("Testing call condition...")

    agent = Agent(
        name="CallAgent",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. When you finish your task, call task_done.",
        tools=[simple_greeting],
        completion_condition="call",
    )

    runner = Runner(agent)
    await runner.run_until_complete("Say hello to Bob and call task_done when finished.")

    print(f"Number of messages: {len(runner.messages)}")

    # Check if task_done was called
    task_done_called = any(
        isinstance(msg, type(runner.messages[0])) and
        hasattr(msg, "name") and
        getattr(msg, "name", "") == "task_done"
        for msg in runner.messages
    )
    print(f"Task done called: {task_done_called}")
    print("Test completed!")


async def main():
    """Run tests."""
    # Set a dummy API key for testing (you should set a real one)
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    await test_stop_condition()
    print()
    await test_call_condition()


if __name__ == "__main__":
    asyncio.run(main())
