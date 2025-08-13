"""
Simple debug to check which API is being used.
"""
import asyncio

from lite_agent import Agent, Runner


async def main():
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="You are helpful.",
    )

    # Test non-streaming with explicit API
    print("Testing non-streaming with responses API...")
    runner = Runner(agent, api="responses", streaming=False)
    print(f"Runner API: {runner.api}")
    print(f"Runner streaming: {runner.streaming}")

    # Test with completion API
    print("\nTesting non-streaming with completion API...")
    runner2 = Runner(agent, api="completion", streaming=False)
    print(f"Runner API: {runner2.api}")
    print(f"Runner streaming: {runner2.streaming}")


if __name__ == "__main__":
    asyncio.run(main())
