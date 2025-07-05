from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import litellm
from litellm.types.llms.openai import ResponsesAPIStreamingResponse
from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel

from lite_agent.loggers import logger
from lite_agent.processors import CompletionEventProcessor, ResponseEventProcessor
from lite_agent.types import AgentChunk

if TYPE_CHECKING:
    from aiofiles.threadpool.text import AsyncTextIOWrapper


def ensure_record_file(record_to: Path | None) -> Path | None:
    if not record_to:
        return None
    if not record_to.parent.exists():
        logger.warning('Record directory "%s" does not exist, creating it.', record_to.parent)
        record_to.parent.mkdir(parents=True, exist_ok=True)
    return record_to


async def litellm_completion_stream_handler(
    resp: litellm.CustomStreamWrapper,
    record_to: Path | None = None,
) -> AsyncGenerator[AgentChunk, None]:
    """
    Optimized chunk handler
    """
    processor = CompletionEventProcessor()
    record_file: AsyncTextIOWrapper | None = None
    record_path = ensure_record_file(record_to)
    if record_path:
        record_file = await aiofiles.open(record_path, "w", encoding="utf-8")
    try:
        async for chunk in resp:  # type: ignore
            if not isinstance(chunk, ModelResponseStream):
                logger.warning("unexpected chunk type: %s", type(chunk))
                logger.warning("chunk content: %s", chunk)
                continue
            async for result in processor.process_chunk(chunk, record_file):
                yield result
    finally:
        if record_file:
            await record_file.close()


async def litellm_response_stream_handler(
    resp: AsyncGenerator[ResponsesAPIStreamingResponse, None],
    record_to: Path | None = None,
) -> AsyncGenerator[AgentChunk, None]:
    """
    Response API stream handler for processing ResponsesAPIStreamingResponse chunks
    """
    processor = ResponseEventProcessor()
    record_file: AsyncTextIOWrapper | None = None
    record_path = ensure_record_file(record_to)
    if record_path:
        record_file = await aiofiles.open(record_path, "w", encoding="utf-8")
    try:
        async for chunk in resp:
            if not isinstance(chunk, BaseModel):
                logger.warning("unexpected chunk type: %s", type(chunk))
                logger.warning("chunk content: %s", chunk)
                continue
            async for result in processor.process_chunk(chunk, record_file):
                yield result
    finally:
        if record_file:
            await record_file.close()
