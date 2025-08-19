"""
Simple debug to check non-streaming response.
"""

import asyncio

from lite_agent import Agent, Runner


async def main():
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="You are helpful.",
    )

    # Test non-streaming
    print("Testing non-streaming...")
    runner = Runner(agent, streaming=False)

    try:
        chunks = []
        async for chunk in runner.run("Say hello"):
            chunks.append(chunk)
            print(f"Received chunk type: {chunk.type}")

        print(f"Total chunks: {len(chunks)}")
        if chunks:
            print(f"First chunk: {chunks[0]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
