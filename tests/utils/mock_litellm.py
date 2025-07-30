"""
Simple mock for litellm acompletion method to facilitate testing.
"""

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import aiofiles
from litellm.types.llms.openai import ResponsesAPIStreamingResponse
from litellm.types.utils import ModelResponseStream
from pydantic import TypeAdapter

from lite_agent.agent import CustomStreamWrapper as RealCustomStreamWrapper


async def mock_acompletion(jsonl_file: str | Path, **_kwargs: object) -> AsyncGenerator[ModelResponseStream, None]:
    """
    Mock litellm acompletion method.

    Args:
        jsonl_file: Path to the jsonl file containing recorded responses
        **_kwargs: Additional arguments (ignored)

    Yields:
        Mock response objects from recorded data
    """
    record_path = Path(jsonl_file)
    if not record_path.exists():
        error_msg = f"No recorded response found at: {record_path}"
        raise FileNotFoundError(error_msg)

    # Read and yield each line as a response object
    async with aiofiles.open(record_path, encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                yield ModelResponseStream.model_validate_json(line)


async def mock_aresponses(jsonl_file: str | Path, **_kwargs: object) -> AsyncGenerator[ResponsesAPIStreamingResponse, None]:
    """
    Mock litellm aresponses method.

    Args:
        jsonl_file: Path to the jsonl file containing recorded responses
        **_kwargs: Additional arguments (ignored)

    Yields:
        Mock response objects from recorded data
    """
    record_path = Path(jsonl_file)
    if not record_path.exists():
        error_msg = f"No recorded response found at: {record_path}"
        raise FileNotFoundError(error_msg)

    # Read and yield each line as a response object
    adapter = TypeAdapter(ResponsesAPIStreamingResponse)
    async with aiofiles.open(record_path, encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                data = json.loads(line)
                yield adapter.validate_python(data)


def create_litellm_mock(jsonl_file: str | Path):
    """
    Create a mock function for litellm.acompletion.

    Args:
        jsonl_file: Path to the jsonl file containing recorded responses

    Returns:
        Mock function that can be used to patch litellm.acompletion

    Example:
        ```python
        from unittest.mock import patch

        mock = create_litellm_mock('recordings/conversation.jsonl')
        with patch('lite_agent.client.litellm.acompletion', mock):
            # Your test code here
            pass
        ```
    """

    class MockCustomStreamWrapper(RealCustomStreamWrapper):
        def __init__(self, async_iterable):
            self._async_iterable = async_iterable

        def __aiter__(self):
            return self._async_iterable.__aiter__()

    async def mock_func(model, messages, **kwargs: object) -> MockCustomStreamWrapper:
        async def _gen() -> AsyncGenerator[ModelResponseStream, None]:
            async for response in mock_acompletion(jsonl_file, **kwargs):
                yield response

        return MockCustomStreamWrapper(_gen())

    return mock_func


def create_litellm_responses_mock(jsonl_file: str | Path):
    """
    Create a mock function for litellm.aresponses.

    Args:
        jsonl_file: Path to the jsonl file containing recorded responses

    Returns:
        Mock function that can be used to patch litellm.aresponses

    Example:
        ```python
        from unittest.mock import patch

        mock = create_litellm_responses_mock('recordings/conversation.jsonl')
        with patch('lite_agent.client.litellm.aresponses', mock):
            # Your test code here
            pass
        ```
    """

    async def mock_func(model=None, input=None, **kwargs: object) -> AsyncGenerator[ResponsesAPIStreamingResponse, None]:
        async for response in mock_aresponses(jsonl_file, **kwargs):
            yield response

    return mock_func
