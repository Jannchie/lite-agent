"""
Demo script showing streaming vs non-streaming configuration in LiteAgent.
"""

import asyncio
import time

from lite_agent import Agent, Runner


async def main():
    # Create an agent
    agent = Agent(
        name="StreamingDemo",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Always respond concisely.",
    )

    print("=== Streaming Mode (Default) ===")
    # Default streaming=True
    runner_streaming = Runner(agent, streaming=True)

    chunks = []
    print("Question: What is the capital of France?")
    print("Response: ", end="", flush=True)
    async for chunk in runner_streaming.run("What is the capital of France?"):
        chunks.append(chunk)
        if chunk.type == "content_delta":
            print(chunk.delta, end="", flush=True)
    print(f"\nReceived {len(chunks)} chunks in streaming mode\n")

    print("=== Non-Streaming Mode ===")
    # Set streaming=False
    runner_non_streaming = Runner(agent, streaming=False)

    chunks = []
    print("Question: What is the capital of Germany?")
    print("Response: ", end="", flush=True)
    async for chunk in runner_non_streaming.run("What is the capital of Germany?"):
        chunks.append(chunk)
        if chunk.type == "assistant_message":
            print(chunk.message.content[0].text)
    print(f"Received {len(chunks)} chunks in non-streaming mode\n")

    print("=== Comparing Performance ===")

    # Time streaming
    start = time.time()
    runner_streaming = Runner(agent, streaming=True)
    chunks = []
    async for chunk in runner_streaming.run("What is 2+2?"):
        chunks.append(chunk)
    streaming_time = time.time() - start

    # Time non-streaming
    start = time.time()
    runner_non_streaming = Runner(agent, streaming=False)
    chunks = []
    async for chunk in runner_non_streaming.run("What is 3+3?"):
        chunks.append(chunk)
    non_streaming_time = time.time() - start

    print(f"Streaming mode: {streaming_time:.2f}s")
    print(f"Non-streaming mode: {non_streaming_time:.2f}s")

    print("\n=== Usage Guide ===")
    print("To use non-streaming mode:")
    print("  runner = Runner(agent, streaming=False)")
    print("To use streaming mode (default):")
    print("  runner = Runner(agent, streaming=True)  # or just Runner(agent)")


if __name__ == "__main__":
    asyncio.run(main())
