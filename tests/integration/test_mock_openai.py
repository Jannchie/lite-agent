"""Integration tests for OpenAI streaming mocks."""

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import pytest

from tests.utils.mock_openai import create_chat_completion_stream_mock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.mark.asyncio
async def test_mock_chat_completion_stream() -> None:
    mock_stream = create_chat_completion_stream_mock("tests/mocks/basic/1.jsonl")
    stream = await mock_stream()
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    assert chunks


@pytest.mark.asyncio
async def test_mock_chat_completion_with_patch() -> None:
    mock_stream = create_chat_completion_stream_mock("tests/mocks/basic/1.jsonl")
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = mock_stream
    mock_client.responses = Mock()

    with patch("lite_agent.client.AsyncOpenAI", return_value=mock_client):
        from lite_agent.client import OpenAIClient

        client = OpenAIClient(model="gpt-4o-mini")
        stream = await client.completion(messages=[{"role": "user", "content": "Hi"}])
        async for _chunk in cast("AsyncIterator[object]", stream):
            pass


@pytest.mark.asyncio
async def test_mock_chat_completion_missing_file() -> None:
    mock_stream = create_chat_completion_stream_mock("tests/mocks/missing.jsonl")

    with pytest.raises(FileNotFoundError):
        await mock_stream()
