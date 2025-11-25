"""
Streaming demo that verifies usage tokens are emitted in streaming mode.
"""

import asyncio
from typing import TYPE_CHECKING

from lite_agent import Agent, Runner

if TYPE_CHECKING:
    from lite_agent.types import AgentChunk, MessageUsage, NewAssistantMessage
    from lite_agent.types.events import Usage


async def main() -> None:
    agent = Agent(
        name="StreamingUsageChecker",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Keep replies short.",
    )

    runner = Runner(agent, streaming=True, api="completion")
    usage_seen: Usage | None = None
    chunks: list[AgentChunk] = []

    print("Prompt: Explain what streaming means in two sentences.")
    print("Streaming response: ", end="", flush=True)

    async for chunk in runner.run("Explain what streaming means in two sentences."):
        chunks.append(chunk)

        if chunk.type == "content_delta":
            print(chunk.delta, end="", flush=True)
        elif chunk.type == "usage":
            usage_seen = chunk.usage

    print("\n\n--- Usage collected during stream ---")
    if usage_seen is not None:
        print(f"Chunk usage -> prompt tokens: {usage_seen.input_tokens}, completion tokens: {usage_seen.output_tokens}")
    else:
        print("No usage chunk received. Check provider support or stream_options.include_usage.")

    print("\n--- Runner aggregated usage ---")
    print(
        f"Runner usage -> prompt tokens: {runner.usage.input_tokens}, completion tokens: {runner.usage.output_tokens}, total: {runner.usage.total_tokens}",
    )

    last_message: NewAssistantMessage | None = None
    for recorded_chunk in reversed(chunks):
        if recorded_chunk.type == "assistant_message":
            last_message = recorded_chunk.message
            break

    print("\n--- Usage attached to assistant message meta ---")
    meta_usage: MessageUsage | None = last_message.meta.usage if last_message is not None else None
    if meta_usage is not None:
        print(
            f"Message meta usage -> prompt tokens: {meta_usage.input_tokens}, completion tokens: {meta_usage.output_tokens}, total: {meta_usage.total_tokens}",
        )
    else:
        print("Assistant message meta has no usage recorded.")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
