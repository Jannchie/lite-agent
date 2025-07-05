"""
测试 litellm_completion_stream_handler 文件记录功能的单元测试
"""

import json
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock

import aiofiles
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

from lite_agent.stream_handlers.litellm import litellm_completion_stream_handler


@pytest.mark.asyncio
async def test_litellm_completion_stream_handler_with_record_to_existing_directory():
    """测试当记录目录存在时的文件记录功能"""
    with tempfile.TemporaryDirectory() as temp_dir:
        record_file = Path(temp_dir) / "test_record.jsonl"

        # 创建模拟的 ModelResponseStream
        mock_stream = ModelResponseStream(
            id="test-123",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4.1-mini",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Hello", role="assistant"),
                    finish_reason=None,
                ),
            ],
        )

        # 创建模拟的响应流
        async def mock_response_stream() -> AsyncGenerator[ModelResponseStream, None]:
            yield mock_stream

        mock_resp = AsyncMock()
        mock_resp.__aiter__ = lambda _: mock_response_stream()

        # 测试 litellm_completion_stream_handler 的文件记录功能
        handler = litellm_completion_stream_handler(mock_resp, record_to=record_file)

        results = []
        async for chunk in handler:
            results.append(chunk)

        # 验证文件被创建
        assert record_file.exists()

        # 验证文件内容
        async with aiofiles.open(record_file, encoding="utf-8") as f:
            content = await f.read()
            lines = content.strip().split("\n")
            assert len(lines) >= 1  # 至少有一行记录
            # 验证每行都是有效的JSON
            for line in lines:
                if line.strip():  # 跳过空行
                    data = json.loads(line.strip())
                    assert "id" in data  # 应该包含记录的数据


@pytest.mark.asyncio
async def test_litellm_completion_stream_handler_with_record_to_nonexistent_directory():
    """测试当记录目录不存在时的自动创建功能"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建一个不存在的子目录路径
        nonexistent_dir = Path(temp_dir) / "nonexistent" / "subdir"
        record_file = nonexistent_dir / "test_record.jsonl"

        # 确保目录不存在
        assert not nonexistent_dir.exists()

        # 创建模拟的 ModelResponseStream
        mock_stream = ModelResponseStream(
            id="test-456",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4.1-mini",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Hello World", role="assistant"),
                    finish_reason="stop",
                ),
            ],
        )

        # 创建模拟的响应流
        async def mock_response_stream() -> AsyncGenerator[ModelResponseStream, None]:
            yield mock_stream

        mock_resp = AsyncMock()
        mock_resp.__aiter__ = lambda _: mock_response_stream()

        # 测试 litellm_completion_stream_handler 的文件记录功能
        handler = litellm_completion_stream_handler(mock_resp, record_to=record_file)

        results = []
        async for chunk in handler:
            results.append(chunk)

        # 验证目录被自动创建
        assert nonexistent_dir.exists()
        assert nonexistent_dir.is_dir()

        # 验证文件被创建
        assert record_file.exists()

        # 验证文件内容
        async with aiofiles.open(record_file, encoding="utf-8") as f:
            content = await f.read()
            lines = content.strip().split("\n")
            assert len(lines) >= 1  # 至少有一行记录


@pytest.mark.asyncio
async def test_litellm_completion_stream_handler_without_record_to():
    """测试不使用文件记录功能时的正常运行"""
    # 创建模拟的 ModelResponseStream
    mock_stream = ModelResponseStream(
        id="test-789",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4.1-mini",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content="No recording", role="assistant"),
                finish_reason="stop",
            ),
        ],
    )

    # 创建模拟的响应流
    async def mock_response_stream() -> AsyncGenerator[ModelResponseStream, None]:
        yield mock_stream

    mock_resp = AsyncMock()
    mock_resp.__aiter__ = lambda self: mock_response_stream()

    # 测试不使用 record_to 参数
    handler = litellm_completion_stream_handler(mock_resp, record_to=None)

    results = []
    async for chunk in handler:
        results.append(chunk)

    # 应该有结果但没有文件被创建
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_litellm_completion_stream_handler_record_multiple_chunks():
    """测试记录多个数据块到文件"""
    with tempfile.TemporaryDirectory() as temp_dir:
        record_file = Path(temp_dir) / "multi_chunk_record.jsonl"

        # 创建多个模拟的 ModelResponseStream
        mock_streams = [
            ModelResponseStream(
                id="test-multi-1",
                object="chat.completion.chunk",
                created=1234567890,
                model="gpt-4.1-mini",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content="First ", role="assistant"),
                        finish_reason=None,
                    ),
                ],
            ),
            ModelResponseStream(
                id="test-multi-2",
                object="chat.completion.chunk",
                created=1234567891,
                model="gpt-4.1-mini",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content="chunk"),
                        finish_reason="stop",
                    ),
                ],
            ),
        ]

        # 创建模拟的响应流
        async def mock_response_stream() -> AsyncGenerator[ModelResponseStream, None]:
            for stream in mock_streams:
                yield stream

        mock_resp = AsyncMock()
        mock_resp.__aiter__ = lambda _: mock_response_stream()

        # 测试 litellm_completion_stream_handler 的文件记录功能
        handler = litellm_completion_stream_handler(mock_resp, record_to=record_file)

        results = []
        async for chunk in handler:
            results.append(chunk)

        # 验证文件被创建
        assert record_file.exists()

        # 验证文件内容 - 应该有多行记录
        async with aiofiles.open(record_file, encoding="utf-8") as f:
            content = await f.read()
            lines = content.strip().split("\n")
            assert len(lines) >= 2  # 至少有两行记录

            # 验证每行都是有效的JSON
            for line in lines:
                if line.strip():
                    data = json.loads(line.strip())
                    # 验证JSON结构包含期望的字段
                    assert "id" in data
                    assert data["id"].startswith("test-multi-")


@pytest.mark.asyncio
async def test_litellm_completion_stream_handler_with_deeply_nested_directory():
    """测试创建深层嵌套目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建一个深层嵌套的目录路径
        deep_dir = Path(temp_dir) / "level1" / "level2" / "level3" / "level4"
        record_file = deep_dir / "deep_record.jsonl"

        # 确保目录不存在
        assert not deep_dir.exists()

        # 创建模拟的 ModelResponseStream
        mock_stream = ModelResponseStream(
            id="test-deep",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4.1-mini",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Deep directory test", role="assistant"),
                    finish_reason="stop",
                ),
            ],
        )

        # 创建模拟的响应流
        async def mock_response_stream() -> AsyncGenerator[ModelResponseStream, None]:
            yield mock_stream

        mock_resp = AsyncMock()
        mock_resp.__aiter__ = lambda _: mock_response_stream()

        # 测试 litellm_completion_stream_handler 的文件记录功能
        handler = litellm_completion_stream_handler(mock_resp, record_to=record_file)

        results = []
        async for chunk in handler:
            results.append(chunk)

        # 验证深层目录被自动创建
        assert deep_dir.exists()
        assert deep_dir.is_dir()

        # 验证文件被创建
        assert record_file.exists()


@pytest.mark.asyncio
async def test_litellm_completion_stream_handler_record_with_string_path():
    """测试使用字符串路径进行文件记录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        record_path_str = str(Path(temp_dir) / "string_path_record.jsonl")

        # 创建模拟的 ModelResponseStream
        mock_stream = ModelResponseStream(
            id="test-string-path",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4.1-mini",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="String path test", role="assistant"),
                    finish_reason="stop",
                ),
            ],
        )

        # 创建模拟的响应流
        async def mock_response_stream() -> AsyncGenerator[ModelResponseStream, None]:
            yield mock_stream

        mock_resp = AsyncMock()
        mock_resp.__aiter__ = lambda _: mock_response_stream()

        # 测试 litellm_completion_stream_handler 的文件记录功能
        handler = litellm_completion_stream_handler(mock_resp, record_to=Path(record_path_str))

        results = []
        async for chunk in handler:
            results.append(chunk)

        # 验证文件被创建
        assert Path(record_path_str).exists()

        # 验证文件内容
        async with aiofiles.open(record_path_str, encoding="utf-8") as f:
            content = await f.read()
            assert content.strip()  # 确保文件不为空
