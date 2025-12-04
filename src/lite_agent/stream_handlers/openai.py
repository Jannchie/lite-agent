import json
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        async for raw_chunk in resp:
            chunk = raw_chunk if isinstance(raw_chunk, ChatCompletionChunk) else _coerce_chat_completion_chunk(raw_chunk)
            if chunk is None:
                logger.warning("unexpected chunk type: %s", type(raw_chunk))
                logger.debug("chunk content: %s", raw_chunk)
                continue
            async for result in processor.process_chunk(chunk, record_file):
                yield result
    finally:
        await _close_stream(resp)
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
        await _close_stream(resp)
        if record_file:
            await record_file.close()


async def _close_stream(stream: object) -> None:
    """Safely close an async stream if it provides an aclose coroutine."""
    close = getattr(stream, "aclose", None)
    if close is None or not callable(close):
        return

    try:
        result = close()
        if isinstance(result, Awaitable):
            await result
    except Exception:
        logger.debug("Failed to close OpenAI stream", exc_info=True)


def _coerce_chat_completion_chunk(chunk: object) -> ChatCompletionChunk | None:
    """Convert objects from LiteLLM/OpenAI into ChatCompletionChunk when possible."""
    if isinstance(chunk, ChatCompletionChunk):
        return chunk

    data: Any | None = None
    if hasattr(chunk, "model_dump"):
        try:
            data = chunk.model_dump()
        except Exception:
            data = None
    elif hasattr(chunk, "model_dump_json"):
        try:
            data = json.loads(chunk.model_dump_json())
        except Exception:
            data = None
    elif isinstance(chunk, dict):
        data = chunk

    if data is None:
        return None

    try:
        return ChatCompletionChunk.model_validate(data)
    except Exception:
        logger.debug("failed to coerce completion chunk", exc_info=True)
        return None
