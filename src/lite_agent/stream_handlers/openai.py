from collections.abc import AsyncGenerator, AsyncIterable
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
from openai._streaming import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import ResponseStreamEvent

from lite_agent.loggers import logger
from lite_agent.processors import CompletionEventProcessor, ResponseEventProcessor
from lite_agent.types import AgentChunk

if TYPE_CHECKING:
    from aiofiles.threadpool.text import AsyncTextIOWrapper


def ensure_record_file(record_to: Path | str | None) -> Path | None:
    if not record_to:
        return None

    path = Path(record_to) if isinstance(record_to, str) else record_to

    # If the path is a directory, generate a filename
    if path.is_dir():
        path = path / "conversation.jsonl"

    # Ensure parent directory exists
    if not path.parent.exists():
        logger.warning('Record directory "%s" does not exist, creating it.', path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


async def openai_completion_stream_handler(
    resp: AsyncStream[ChatCompletionChunk] | AsyncIterable[ChatCompletionChunk],
    record_to: Path | str | None = None,
) -> AsyncGenerator[AgentChunk, None]:
    """Process streaming Chat Completions from the OpenAI SDK."""

    processor = CompletionEventProcessor()
    record_file: AsyncTextIOWrapper | None = None
    record_path = ensure_record_file(record_to)
    if record_path:
        record_file = await aiofiles.open(record_path, "w", encoding="utf-8")

    try:
        async for chunk in resp:
            if not isinstance(chunk, ChatCompletionChunk):
                logger.warning("unexpected chunk type: %s", type(chunk))
                logger.warning("chunk content: %s", chunk)
                continue
            async for result in processor.process_chunk(chunk, record_file):
                yield result
    finally:
        if hasattr(resp, "aclose") and callable(resp.aclose):
            try:
                await resp.aclose()  # type: ignore[func-returns-value]
            except Exception:
                logger.debug("Failed to close OpenAI stream", exc_info=True)
        if record_file:
            await record_file.close()


async def openai_response_stream_handler(
    resp: AsyncStream[ResponseStreamEvent] | AsyncIterable[ResponseStreamEvent],
    record_to: Path | str | None = None,
) -> AsyncGenerator[AgentChunk, None]:
    """Process streaming Responses API events from the OpenAI SDK."""

    processor = ResponseEventProcessor()
    record_file: AsyncTextIOWrapper | None = None
    record_path = ensure_record_file(record_to)
    if record_path:
        record_file = await aiofiles.open(record_path, "w", encoding="utf-8")

    try:
        async for chunk in resp:
            if not hasattr(chunk, "model_dump_json"):
                logger.warning("unexpected chunk type: %s", type(chunk))
                logger.warning("chunk content: %s", chunk)
                continue
            async for result in processor.process_chunk(chunk, record_file):
                yield result
    finally:
        if hasattr(resp, "aclose") and callable(resp.aclose):
            try:
                await resp.aclose()  # type: ignore[func-returns-value]
            except Exception:
                logger.debug("Failed to close OpenAI stream", exc_info=True)
        if record_file:
            await record_file.close()
