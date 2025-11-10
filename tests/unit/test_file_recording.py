"""Tests for recording functionality of OpenAI completion stream handler."""

import json
import tempfile
from pathlib import Path
from typing import Any

import aiofiles
import pytest
from openai.types.chat import ChatCompletionChunk

from lite_agent.stream_handlers.openai import openai_completion_stream_handler


class MockAsyncStream:
    """Simple async iterable to simulate OpenAI streaming responses."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item

        return gen()

    async def aclose(self) -> None:
        return


def _build_chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk-id",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content,
                    },
                },
            ],
        },
    )


@pytest.mark.asyncio
async def test_completion_stream_handler_with_existing_directory() -> None:
    """Streaming handler should record chunks when directory exists."""

    with tempfile.TemporaryDirectory() as temp_dir:
        record_file = Path(temp_dir) / "test_record.jsonl"
        stream = MockAsyncStream([_build_chunk("Hello")])

        results = []
        async for chunk in openai_completion_stream_handler(stream, record_to=record_file):
            results.append(chunk)

        assert record_file.exists()
        async with aiofiles.open(record_file, encoding="utf-8") as f:
            content = await f.read()
        lines = [line for line in content.splitlines() if line.strip()]
        assert lines
        for line in lines:
            data = json.loads(line)
            assert data["model"] == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_completion_stream_handler_creates_directory() -> None:
    """Streaming handler should auto-create directories when needed."""

    with tempfile.TemporaryDirectory() as temp_dir:
        record_dir = Path(temp_dir) / "nested" / "dir"
        record_file = record_dir / "record.jsonl"
        assert not record_dir.exists()

        stream = MockAsyncStream([_build_chunk("World")])

        async for _chunk in openai_completion_stream_handler(stream, record_to=record_file):
            pass

        assert record_dir.exists()
        assert record_file.exists()


@pytest.mark.asyncio
async def test_completion_stream_handler_without_record_file() -> None:
    """Handler should work without recording to file."""

    stream = MockAsyncStream([_build_chunk("No recording")])

    results = []
    async for chunk in openai_completion_stream_handler(stream, record_to=None):
        results.append(chunk)

    assert results


@pytest.mark.asyncio
async def test_completion_stream_handler_records_multiple_chunks() -> None:
    """Multiple chunks should be appended to the record file."""

    with tempfile.TemporaryDirectory() as temp_dir:
        record_file = Path(temp_dir) / "multiple.jsonl"
        stream = MockAsyncStream([
            _build_chunk("First"),
            _build_chunk("Second"),
        ])

        async for _chunk in openai_completion_stream_handler(stream, record_to=record_file):
            pass

        async with aiofiles.open(record_file, encoding="utf-8") as f:
            lines = [line for line in (await f.read()).splitlines() if line.strip()]

        assert len(lines) == 2
