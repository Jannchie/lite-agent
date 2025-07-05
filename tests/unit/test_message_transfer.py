"""Test the message_transfer callback functionality."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

from lite_agent.agent import Agent
from lite_agent.types import AgentChunk, RunnerMessages


def test_agent_initialization_with_message_transfer():
    """Test that agents can be initialized with a message_transfer callback."""

    def dummy_transfer(messages: RunnerMessages) -> RunnerMessages:
        return messages

    agent = Agent(
        model="gpt-4.1-mini",
        name="TestAgent",
        instructions="Test instructions",
        message_transfer=dummy_transfer,
    )

    assert agent.message_transfer is not None
    assert agent.message_transfer == dummy_transfer


def test_agent_initialization_without_message_transfer():
    """Test that agents can be initialized without a message_transfer callback."""
    agent = Agent(
        model="gpt-4.1-mini",
        name="TestAgent",
        instructions="Test instructions",
    )

    assert agent.message_transfer is None


def test_set_message_transfer():
    """Test that message_transfer can be set after initialization."""

    def dummy_transfer(messages: RunnerMessages) -> RunnerMessages:
        return messages

    agent = Agent(
        model="gpt-4.1-mini",
        name="TestAgent",
        instructions="Test instructions",
    )

    assert agent.message_transfer is None

    agent.set_message_transfer(dummy_transfer)
    assert agent.message_transfer == dummy_transfer

    # Test setting to None
    agent.set_message_transfer(None)
    assert agent.message_transfer is None


@pytest.mark.asyncio
async def test_message_transfer_called_in_completion():
    """Test that message_transfer callback is called during completion."""

    def test_transfer(messages: RunnerMessages) -> RunnerMessages:
        """Add a prefix to all user messages."""
        processed = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                new_msg = msg.copy()
                new_msg["content"] = f"[TEST] {msg['content']}"  # type: ignore
                processed.append(new_msg)
            else:
                processed.append(msg)
        return processed

    agent = Agent(
        model="gpt-3.5-turbo",
        name="TestAgent",
        instructions="Test instructions",
        message_transfer=test_transfer,
    )

    test_messages = [{"role": "user", "content": "Hello world"}]

    # Mock the litellm.acompletion to avoid actual API calls
    from litellm import CustomStreamWrapper

    mock_resp = AsyncMock(spec=CustomStreamWrapper)

    async def mock_stream_handler(*_args, **_kwargs) -> AsyncGenerator[AgentChunk, None]:  # type: ignore[misc]
        """Mock async generator for stream handler."""
        return
        yield  # pragma: no cover

    with patch("lite_agent.client.litellm.acompletion") as mock_completion, patch("lite_agent.agent.litellm_completion_stream_handler", new=mock_stream_handler):
        mock_completion.return_value = mock_resp

        # Call completion
        async for _ in await agent.completion(test_messages):
            pass

        # Verify that completion was called
        mock_completion.assert_called_once()

        # Get the messages that were passed to the completion
        call_args = mock_completion.call_args
        passed_messages = call_args[1]["messages"]

        # The first message should be the system message
        # The second message should be our processed user message
        assert len(passed_messages) >= 2
        user_message = passed_messages[1]  # Skip system message
        assert user_message["content"] == "[TEST] Hello world"


@pytest.mark.asyncio
async def test_completion_without_message_transfer():
    """Test that completion works normally without message_transfer callback."""
    agent = Agent(
        model="gpt-3.5-turbo",
        name="TestAgent",
        instructions="Test instructions",
    )

    test_messages = [{"role": "user", "content": "Hello world"}]

    # Mock the litellm.acompletion to avoid actual API calls
    from litellm import CustomStreamWrapper

    mock_resp = AsyncMock(spec=CustomStreamWrapper)

    async def mock_stream_handler(*_args, **_kwargs) -> AsyncGenerator[AgentChunk, None]:  # type: ignore[misc]
        """Mock async generator for stream handler."""
        return
        yield  # pragma: no cover

    with patch("lite_agent.client.litellm.acompletion") as mock_completion, patch("lite_agent.agent.litellm_completion_stream_handler", new=mock_stream_handler):
        mock_completion.return_value = mock_resp

        # Call completion
        async for _ in await agent.completion(test_messages):
            pass

        # Verify that completion was called
        mock_completion.assert_called_once()

        # Get the messages that were passed to the completion
        call_args = mock_completion.call_args
        passed_messages = call_args[1]["messages"]

        # The first message should be the system message
        # The second message should be our original user message (unchanged)
        assert len(passed_messages) >= 2
        user_message = passed_messages[1]  # Skip system message
        assert user_message["content"] == "Hello world"


def test_message_transfer_with_different_message_types():
    """Test message_transfer with different message types."""

    def add_prefix_transfer(messages: RunnerMessages) -> RunnerMessages:
        """Add prefix to user messages only."""
        processed = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "user" and "content" in msg:
                    new_msg = msg.copy()
                    new_msg["content"] = f"[USER] {msg['content']}"
                    processed.append(new_msg)
                elif msg.get("role") == "assistant" and "content" in msg:
                    new_msg = msg.copy()
                    new_msg["content"] = f"[ASSISTANT] {msg['content']}"
                    processed.append(new_msg)
                else:
                    processed.append(msg)
            else:
                processed.append(msg)
        return processed

    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "system", "content": "System message"},
        {"type": "function_call", "name": "test_func", "arguments": "{}"},
    ]

    result = add_prefix_transfer(test_messages)

    assert result[0]["content"] == "[USER] Hello"  # type: ignore
    assert result[1]["content"] == "[ASSISTANT] Hi there"  # type: ignore
    assert result[2]["content"] == "System message"  # type: ignore # Unchanged
    assert result[3] == test_messages[3]  # Unchanged function call
