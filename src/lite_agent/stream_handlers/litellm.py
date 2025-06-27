from collections.abc import AsyncGenerator
from pathlib import Path

import aiofiles
import litellm
from aiofiles.threadpool.text import AsyncTextIOWrapper
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

from lite_agent.loggers import logger
from lite_agent.processors import StreamChunkProcessor
from lite_agent.types import AgentChunk, CompletionRawChunk, ContentDeltaChunk, FinalMessageChunk, ToolCallDeltaChunk, UsageChunk


def ensure_record_file(record_to: Path | None) -> Path | None:
    if not record_to:
        return None
    if not record_to.parent.exists():
        logger.warning('Record directory "%s" does not exist, creating it.', record_to.parent)
        record_to.parent.mkdir(parents=True, exist_ok=True)
    return record_to


async def process_chunk(
    processor: StreamChunkProcessor,
    chunk: ModelResponseStream,
    record_file: AsyncTextIOWrapper | None = None,
) -> AsyncGenerator[AgentChunk, None]:
    if record_file:
        await record_file.write(chunk.model_dump_json() + "\n")
        await record_file.flush()
    yield CompletionRawChunk(type="completion_raw", raw=chunk)
    usage_chunk = await handle_usage_chunk(processor, chunk)
    if usage_chunk:
        yield usage_chunk
        return
    if not chunk.choices:
        return
    choice = chunk.choices[0]
    delta = choice.delta
    for result in await handle_content_and_tool_calls(processor, chunk, choice, delta):
        yield result
    if choice.finish_reason:
        current_message = processor.current_message
        yield FinalMessageChunk(type="final_message", message=current_message, finish_reason=choice.finish_reason)


async def handle_usage_chunk(processor: StreamChunkProcessor, chunk: ModelResponseStream) -> UsageChunk | None:
    usage = processor.handle_usage_info(chunk)
    if usage:
        return UsageChunk(type="usage", usage=usage)
    return None


async def handle_content_and_tool_calls(
    processor: StreamChunkProcessor,
    chunk: ModelResponseStream,
    choice: StreamingChoices,
    delta: Delta,
) -> list[AgentChunk]:
    results: list[AgentChunk] = []
    if not processor.is_initialized:
        processor.initialize_message(chunk, choice)
    if delta.content:
        results.append(ContentDeltaChunk(type="content_delta", delta=delta.content))
        processor.update_content(delta.content)
    if delta.tool_calls is not None:
        processor.update_tool_calls(delta.tool_calls)
        if delta.tool_calls and processor.current_message.tool_calls:
            results.extend(
                [
                    ToolCallDeltaChunk(
                        type="tool_call_delta",
                        tool_call_id=processor.current_message.tool_calls[-1].id,
                        name=processor.current_message.tool_calls[-1].function.name,
                        arguments_delta=tool_call.function.arguments or "",
                    )
                    for tool_call in delta.tool_calls
                    if tool_call.function.arguments
                ],
            )
    return results


async def litellm_stream_handler(
    resp: litellm.CustomStreamWrapper,
    record_to: Path | None = None,
) -> AsyncGenerator[AgentChunk, None]:
    """
    Optimized chunk handler
    """
    processor = StreamChunkProcessor()
    record_file: AsyncTextIOWrapper | None = None
    record_path = ensure_record_file(record_to)
    if record_path:
        record_file = await aiofiles.open(record_path, "a", encoding="utf-8")
    try:
        async for chunk in resp:  # type: ignore
            if not isinstance(chunk, ModelResponseStream):
                logger.warning("unexpected chunk type: %s", type(chunk))
                logger.warning("chunk content: %s", chunk)
                continue
            async for result in process_chunk(processor, chunk, record_file):
                yield result
    finally:
        if record_file:
            await record_file.close()
