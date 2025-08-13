"""
Debug example to investigate non-streaming mode issues.
"""
import asyncio
import logging

from lite_agent import Agent, Runner

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


async def main():
    # Create an agent
    agent = Agent(
        name="DebugAgent",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
    )

    print("=== Debug Non-Streaming Mode ===")

    # Test with streaming=False
    runner = Runner(agent, streaming=False)

    print("Running in non-streaming mode...")
    chunks = []

    async for chunk in runner.run("Hello, please say hi back."):
        print(f"Received chunk: {chunk}")
        print(f"Chunk type: {chunk.type}")
        if hasattr(chunk, "message"):
            print(f"Chunk message: {chunk.message}")
        if hasattr(chunk, "content"):
            print(f"Chunk content: {chunk.content}")
        chunks.append(chunk)

    print(f"\nTotal chunks received: {len(chunks)}")

    # Compare with streaming mode
    print("\n=== Compare with Streaming Mode ===")
    runner_streaming = Runner(agent, streaming=True)

    streaming_chunks = []
    async for chunk in runner_streaming.run("Hello, please say hi back too."):
        streaming_chunks.append(chunk)
        if chunk.type == "content_delta":
            print(chunk.delta, end="", flush=True)

    print(f"\nStreaming chunks received: {len(streaming_chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
