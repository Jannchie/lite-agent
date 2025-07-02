"""
Example test showing how to use the litellm mock.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.utils.mock_litellm import create_litellm_mock


@pytest.mark.asyncio
async def test_mock_litellm_basic():
    """Test basic functionality of the litellm mock."""
    # Create a temporary directory and test data
    with tempfile.TemporaryDirectory() as temp_dir:
        record_dir = Path(temp_dir)

        # Create mock response file
        mock_responses = [
            {"id": "test-1", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "test-2", "choices": [{"delta": {"content": " there!"}}]},
            {"id": "test-3", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 10}},
        ]

        response_file = record_dir / "test_responses.jsonl"
        with response_file.open("w", encoding="utf-8") as f:
            for response in mock_responses:
                f.write(json.dumps(response) + "\n")

        # Create mock and test - now pass the jsonl file directly
        mock = create_litellm_mock(response_file)

        responses = []
        stream = await mock(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
        async for response in stream:
            responses.append(response)

        # Verify responses
        assert len(responses) == 3
        assert responses[0].id == "test-1"
        assert responses[0].choices[0].delta.content == "Hello"
        assert responses[1].choices[0].delta.content == " there!"
        assert responses[2].choices[0].finish_reason == "stop"
        assert responses[2].usage.total_tokens == 10


@pytest.mark.asyncio
async def test_mock_litellm_file_not_found():
    """Test behavior when recorded file is not found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        record_dir = Path(temp_dir)

        # Use a non-existent file
        non_existent_file = record_dir / "nonexistent.jsonl"
        mock = create_litellm_mock(non_existent_file)

        # Test that FileNotFoundError is raised when iterating
        async def iterate_mock() -> None:
            stream = await mock(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
            async for _ in stream:
                pass

        with pytest.raises(FileNotFoundError):
            await iterate_mock()


@pytest.mark.asyncio
async def test_mock_litellm_with_patch():
    """Test using the mock with unittest.mock.patch."""
    with tempfile.TemporaryDirectory() as temp_dir:
        record_dir = Path(temp_dir)

        # Create mock response file
        mock_response = {"id": "patch-test", "choices": [{"delta": {"content": "Mocked response"}}]}
        response_file = record_dir / "patch_test.jsonl"
        with response_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(mock_response) + "\n")

        # Create mock
        mock = create_litellm_mock(response_file)

        # Test with patch (this would be used in actual tests of your agent)
        with patch("lite_agent.client.litellm.acompletion", mock):
            # Simulate what your agent would do
            stream = await mock(model="gpt-4", messages=[{"role": "user", "content": "Test"}])
            async for response in stream:
                assert response.id == "patch-test"
                assert response.choices[0].delta.content == "Mocked response"
                break
