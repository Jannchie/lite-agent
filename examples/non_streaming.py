"""
Simple example demonstrating non-streaming mode in LiteAgent.
"""

import asyncio
from datetime import datetime, timezone

from lite_agent import Agent, Runner


async def main():
    # Create an agent
    agent = Agent(
        name="NonStreamingDemo",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
    )

    # Create runner with non-streaming mode
    runner = Runner(agent, streaming=False)

    print("=== Non-Streaming Mode Example ===")
    print("Question: Explain what Python is in one sentence.")
    print("Response: ", end="", flush=True)

    # In non-streaming mode, you get the complete response at once
    async for chunk in runner.run("Explain what Python is in one sentence."):
        if chunk.type == "assistant_message":
            # Non-streaming mode typically yields one complete message
            print(chunk.message.content[0].text)

    print("\n=== Tool Usage with Non-Streaming ===")

    # Example with a simple tool
    def get_time() -> str:
        """Get the current time."""
        return f"Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"

    agent_with_tools = Agent(
        name="TimeAgent",
        model="gpt-4o-mini",
        instructions="You help users with time-related queries. Use the get_time tool when asked about time.",
        tools=[get_time],
    )

    runner_with_tools = Runner(agent_with_tools, streaming=False)

    print("Question: What time is it now?")
    print("Response:")

    async for chunk in runner_with_tools.run("What time is it now?"):
        if chunk.type == "assistant_message":
            print(chunk.message.content[0].text)
        elif chunk.type == "function_call_output":
            print(f"Tool output: {chunk.content}")

    print("\n=== Benefits of Non-Streaming Mode ===")
    print("1. Simpler processing - get complete responses")
    print("2. Easier for batch processing")
    print("3. Better for APIs that need complete responses")
    print("4. Lower overhead for short interactions")


if __name__ == "__main__":
    asyncio.run(main())
