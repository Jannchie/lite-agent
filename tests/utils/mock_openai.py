"""Utilities to mock OpenAI async streaming methods using recorded JSONL files."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any

import aiofiles
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import ResponseStreamEvent


class MockAsyncStream:
    """Simple async stream wrapper mimicking OpenAI AsyncStream."""

    def __init__(self, factory: Callable[[], AsyncGenerator[Any, None]]) -> None:
        self._factory = factory

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._factory()

    async def aclose(self) -> None:
        return None


async def _read_jsonl(file_path: Path) -> AsyncGenerator[dict, None]:
    async with aiofiles.open(file_path, encoding="utf-8") as handle:
        async for line in handle:
            if line.strip():
                yield json.loads(line)


def create_chat_completion_stream_mock(jsonl_file: str | Path):
    """Return an async function mocking AsyncOpenAI.chat.completions.create."""

    record_path = Path(jsonl_file)

    async def _mock(*_args, **_kwargs) -> MockAsyncStream:
        if not record_path.exists():
            msg = f"No recorded response found at: {record_path}"
            raise FileNotFoundError(msg)

        async def iterator() -> AsyncGenerator[ChatCompletionChunk, None]:
            async for payload in _read_jsonl(record_path):
                yield ChatCompletionChunk.model_validate(payload)

        return MockAsyncStream(iterator)

    return _mock


def create_responses_stream_mock(jsonl_file: str | Path):
    """Return an async function mocking AsyncOpenAI.responses.create."""

    record_path = Path(jsonl_file)

    async def _mock(*_args, **_kwargs) -> MockAsyncStream:
        if not record_path.exists():
            msg = f"No recorded response found at: {record_path}"
            raise FileNotFoundError(msg)

        async def iterator() -> AsyncGenerator[ResponseStreamEvent, None]:
            async for payload in _read_jsonl(record_path):
                yield ResponseStreamEvent.model_validate(payload)

        return MockAsyncStream(iterator)

    return _mock
