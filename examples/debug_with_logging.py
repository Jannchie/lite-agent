"""
Debug with full logging enabled.
"""

import asyncio
import logging

from lite_agent import Agent, Runner

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
)

# Enable specific loggers
logging.getLogger("lite_agent").setLevel(logging.DEBUG)


async def main():
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="You are helpful.",
    )

    print("=== Testing Non-Streaming ===")
    runner = Runner(agent, streaming=False)

    chunks = []
    async for chunk in runner.run("Hello"):
        chunks.append(chunk)
        print(f"Got chunk: {chunk.type}")

    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
