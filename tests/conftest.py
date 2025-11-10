"""Shared pytest fixtures for the test suite."""

from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_async_openai() -> Generator[Mock, None, None]:
    """Provide a dummy AsyncOpenAI client for all tests."""

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = AsyncMock(return_value=Mock())
    mock_client.responses = Mock()
    mock_client.responses.create = AsyncMock(return_value=Mock())

    with patch("lite_agent.client.AsyncOpenAI", return_value=mock_client):
        yield mock_client
