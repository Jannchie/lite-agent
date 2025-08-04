"""
Additional tests for stream_handlers/litellm.py to improve coverage
"""

import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import litellm
import pytest
from litellm.types.llms.openai import ResponsesAPIStreamingResponse

from lite_agent.stream_handlers.litellm import (
    ensure_record_file,
    litellm_completion_stream_handler,
    litellm_response_stream_handler,
)


class TestStreamHandlersAdditional:
    """Additional tests for litellm stream handlers"""

    def test_ensure_record_file_with_none(self):
        """Test ensure_record_file with None input"""
        result = ensure_record_file(None)
        assert result is None

    def test_ensure_record_file_with_path(self):
        """Test ensure_record_file with Path input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = ensure_record_file(temp_path)
            assert result is not None
            assert result.parent == temp_path

    def test_ensure_record_file_with_string_path(self):
        """Test ensure_record_file with string input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ensure_record_file(temp_dir)
            assert result is not None
            assert result.parent == Path(temp_dir)

    def test_ensure_record_file_creates_directory(self):
        """Test ensure_record_file creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = Path(temp_dir) / "non_existent"
            result = ensure_record_file(non_existent_dir)
            assert result is not None
            assert result.parent.exists()

    @pytest.mark.asyncio
    async def test_litellm_completion_stream_handler_unexpected_chunk(self):
        """Test litellm_completion_stream_handler with unexpected chunk type"""
        # Create a mock chunk that's not a BaseModel
        unexpected_chunk = "not a basemodel"

        async def mock_async_iter() -> AsyncGenerator[str, None]:
            yield unexpected_chunk

        mock_resp = Mock(spec=litellm.CustomStreamWrapper)
        mock_resp.__aiter__ = Mock(return_value=mock_async_iter())

        with patch("lite_agent.stream_handlers.litellm.logger") as mock_logger:
            chunks = []
            async for chunk in litellm_completion_stream_handler(mock_resp):
                chunks.append(chunk)

            # Should log warning about unexpected chunk type
            assert mock_logger.warning.call_count >= 1

    @pytest.mark.asyncio
    async def test_litellm_completion_stream_handler_with_record_file(self):
        """Test litellm_completion_stream_handler with record file"""
        from pydantic import BaseModel

        class MockChunk(BaseModel):
            id: str = "test"
            content: str = "test content"

        chunk = MockChunk()

        async def mock_async_iter() -> AsyncGenerator[Any, None]:
            yield chunk

        mock_resp = Mock(spec=litellm.CustomStreamWrapper)
        mock_resp.__aiter__ = Mock(return_value=mock_async_iter())

        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = Path(temp_dir)

            with patch("lite_agent.stream_handlers.litellm.aiofiles") as mock_aiofiles:
                mock_file = AsyncMock()
                mock_aiofiles.open = AsyncMock(return_value=mock_file)

                with patch("lite_agent.stream_handlers.litellm.CompletionEventProcessor") as mock_processor:
                    mock_processor_instance = Mock()

                    # Create proper async generator mock
                    async def mock_process_chunk(_chunk: object, _record_file: object) -> AsyncGenerator[None, None]:
                        return
                        yield  # unreachable but makes it an async generator

                    mock_processor_instance.process_chunk = mock_process_chunk
                    mock_processor.return_value = mock_processor_instance

                    chunks = []
                    async for chunk in litellm_completion_stream_handler(mock_resp, record_to=record_path):
                        chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_litellm_response_stream_handler_unexpected_chunk(self):
        """Test litellm_response_stream_handler with unexpected chunk type"""
        # Create a mock chunk that's not a proper ResponsesAPIStreamingResponse
        async def mock_resp() -> AsyncGenerator[ResponsesAPIStreamingResponse, None]:
            # Use type: ignore to bypass type checking for testing purposes
            yield "not a basemodel"  # type: ignore[misc]

        with patch("lite_agent.stream_handlers.litellm.logger") as mock_logger:
            chunks = []
            async for chunk in litellm_response_stream_handler(mock_resp()):
                chunks.append(chunk)

            # Should log warning about unexpected chunk type
            assert mock_logger.warning.call_count >= 1

    @pytest.mark.asyncio
    async def test_litellm_response_stream_handler_with_record_file(self):
        """Test litellm_response_stream_handler with record file"""
        from pydantic import BaseModel

        class MockResponse(BaseModel):
            id: str = "test"
            content: str = "test content"

        response = MockResponse()

        async def mock_resp() -> AsyncGenerator[ResponsesAPIStreamingResponse, None]:
            # Use type: ignore to bypass type checking for testing purposes
            yield response  # type: ignore[misc]

        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = Path(temp_dir)

            with patch("lite_agent.stream_handlers.litellm.aiofiles") as mock_aiofiles:
                mock_file = AsyncMock()
                mock_aiofiles.open = AsyncMock(return_value=mock_file)

                with patch("lite_agent.stream_handlers.litellm.ResponseEventProcessor") as mock_processor:
                    mock_processor_instance = Mock()

                    # Create proper async generator mock
                    async def mock_process_chunk(_chunk: object, _record_file: object) -> AsyncGenerator[None, None]:
                        return
                        yield  # unreachable but makes it an async generator

                    mock_processor_instance.process_chunk = mock_process_chunk
                    mock_processor.return_value = mock_processor_instance

                    chunks = []
                    async for chunk in litellm_response_stream_handler(mock_resp(), record_to=record_path):
                        chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_stream_handlers_file_operations(self):
        """Test file operations in stream handlers"""
        from pydantic import BaseModel

        class MockChunk(BaseModel):
            id: str = "test"

        chunk = MockChunk()

        async def mock_async_iter() -> AsyncGenerator[Any, None]:
            yield chunk

        mock_resp = Mock(spec=litellm.CustomStreamWrapper)
        mock_resp.__aiter__ = Mock(return_value=mock_async_iter())

        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = Path(temp_dir) / "test.jsonl"

            with patch("lite_agent.stream_handlers.litellm.aiofiles") as mock_aiofiles:
                mock_file = AsyncMock()
                mock_aiofiles.open = AsyncMock(return_value=mock_file)

                with patch("lite_agent.stream_handlers.litellm.CompletionEventProcessor") as mock_processor:
                    mock_processor_instance = Mock()

                    # Mock the async generator
                    async def mock_process_chunk(_chunk: object, record_file: Any) -> AsyncGenerator[None, None]:  # noqa: ANN401
                        if record_file:
                            await record_file.write('{"test": "data"}\n')
                        return
                        yield  # Make it an async generator

                    mock_processor_instance.process_chunk = mock_process_chunk
                    mock_processor.return_value = mock_processor_instance

                    chunks = []
                    async for chunk in litellm_completion_stream_handler(mock_resp, record_to=record_path.parent):
                        chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_stream_handlers_exception_handling(self):
        """Test exception handling in stream handlers"""

        class TestError(Exception):
            pass

        async def failing_async_iter() -> AsyncGenerator[Any, None]:
            test_exception_msg = "Test exception"
            raise TestError(test_exception_msg)
            yield  # Make it an async generator

        mock_resp = Mock(spec=litellm.CustomStreamWrapper)
        mock_resp.__aiter__ = Mock(return_value=failing_async_iter())

        with pytest.raises(TestError, match="Test exception"):
            async for _chunk in litellm_completion_stream_handler(mock_resp):
                pass

    @pytest.mark.asyncio
    async def test_record_file_cleanup(self):
        """Test that record files are properly closed"""
        from pydantic import BaseModel

        class MockChunk(BaseModel):
            id: str = "test"

        async def mock_async_iter() -> AsyncGenerator[Any, None]:
            yield MockChunk()

        mock_resp = Mock(spec=litellm.CustomStreamWrapper)
        mock_resp.__aiter__ = Mock(return_value=mock_async_iter())

        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = Path(temp_dir)

            with patch("lite_agent.stream_handlers.litellm.aiofiles") as mock_aiofiles:
                mock_file = AsyncMock()
                # Make aiofiles.open() return the mock file when awaited
                mock_aiofiles.open = AsyncMock(return_value=mock_file)

                with patch("lite_agent.stream_handlers.litellm.CompletionEventProcessor") as mock_processor:
                    mock_processor_instance = Mock()

                    # Create a proper async generator mock
                    async def mock_process_chunk(_chunk: object, _record_file: object) -> AsyncGenerator[None, None]:
                        return
                        yield  # unreachable but makes it an async generator

                    mock_processor_instance.process_chunk = mock_process_chunk
                    mock_processor.return_value = mock_processor_instance

                    chunks = []
                    async for chunk in litellm_completion_stream_handler(mock_resp, record_to=record_path):
                        chunks.append(chunk)

                    # Verify file was closed
                    mock_file.close.assert_called_once()
