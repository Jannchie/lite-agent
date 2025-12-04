"""Additional tests for OpenAI stream handlers."""

import json
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai.types.chat import ChatCompletionChunk

from lite_agent.stream_handlers.openai import (
    ensure_record_file,
    openai_completion_stream_handler,
    openai_response_stream_handler,
)


class MockAsyncStream:
    """Simple async iterable to simulate OpenAI streams."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._closed = False

    def __aiter__(self) -> AsyncGenerator[Any, None]:
        async def gen() -> AsyncGenerator[Any, None]:
            for item in self._items:
                yield item

        return gen()

    async def aclose(self) -> None:
        self._closed = True


class DummyEvent:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    def model_dump_json(self) -> str:
        return json.dumps(self.__dict__)


def _build_chat_chunk(delta: dict[str, Any]) -> ChatCompletionChunk:
    """Create a minimal valid ChatCompletionChunk for testing."""

    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk-id",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                },
            ],
        },
    )


class TestStreamHandlersAdditional:
    """Additional tests for OpenAI stream handlers."""

    def test_ensure_record_file_variants(self) -> None:
        """ensure_record_file should handle None, Path, and string inputs."""

        assert ensure_record_file(None) is None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            expected_file = temp_path / "conversation.jsonl"
            assert ensure_record_file(temp_path) == expected_file
            assert ensure_record_file(temp_dir) == expected_file

            nested = temp_path / "subdir" / "file.jsonl"
            result = ensure_record_file(nested)
            assert result == nested
            assert nested.parent.exists()

    @pytest.mark.asyncio
    async def test_completion_stream_handler_logs_unexpected_chunk(self) -> None:
        """Non-ChatCompletionChunk inputs should trigger a warning and be skipped."""

        stream = MockAsyncStream(["invalid"])

        with patch("lite_agent.stream_handlers.openai.logger") as mock_logger:
            collected = [chunk async for chunk in openai_completion_stream_handler(stream)]

        assert collected == []
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_completion_stream_handler_with_record_file(self, tmp_path: Path) -> None:
        """Handler should open record file and delegate to processor."""

        chunk = _build_chat_chunk({"role": "assistant", "content": "hi"})
        stream = MockAsyncStream([chunk])

        mock_file = AsyncMock()
        mock_file.write = AsyncMock()
        mock_file.flush = AsyncMock()
        mock_file.close = AsyncMock()

        mock_open = AsyncMock(return_value=mock_file)

        with patch("lite_agent.stream_handlers.openai.CompletionEventProcessor") as mock_processor_cls, patch(
            "lite_agent.stream_handlers.openai.aiofiles.open",
            mock_open,
        ):
            mock_processor_instance = Mock()
            async def mock_process_chunk(*_args: object, **_kwargs: object) -> AsyncGenerator[None, None]:
                if False:
                    yield None

            mock_processor_instance.process_chunk = mock_process_chunk
            mock_processor_cls.return_value = mock_processor_instance

            async for _chunk in openai_completion_stream_handler(stream, tmp_path / "record.jsonl"):
                pass

        mock_processor_cls.assert_called_once()
        mock_open.assert_awaited_once()
        mock_file.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_response_stream_handler_unexpected_chunk(self) -> None:
        """Non-BaseModel events should be ignored with warning."""

        stream = MockAsyncStream(["invalid"])

        with patch("lite_agent.stream_handlers.openai.logger") as mock_logger:
            collected = [chunk async for chunk in openai_response_stream_handler(stream)]

        assert collected == []
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_response_stream_handler_processes_events(self) -> None:
        """Handler should forward events to ResponseEventProcessor."""

        output_added = DummyEvent(
            type="response.output_item.added",
            item={"type": "message", "content": []},
            output_index=0,
            sequence_number=0,
        )
        text_delta = DummyEvent(
            type="response.output_text.delta",
            delta="Hello",
            item_id="item",
            output_index=0,
            content_index=0,
            sequence_number=1,
        )
        output_done = DummyEvent(
            type="response.output_item.done",
            item={"type": "function_call", "call_id": "1", "name": "tool", "arguments": "{}"},
            output_index=0,
            sequence_number=2,
        )
        stream = MockAsyncStream([output_added, text_delta, output_done])

        with patch("lite_agent.stream_handlers.openai.ResponseEventProcessor") as mock_processor_cls:
            processor_instance = Mock()
            processor_instance.process_chunk = Mock(return_value=MockAsyncStream([]))
            mock_processor_cls.return_value = processor_instance

            async for _chunk in openai_response_stream_handler(stream):
                pass

        mock_processor_cls.assert_called_once()
        assert processor_instance.process_chunk.call_count == 3
