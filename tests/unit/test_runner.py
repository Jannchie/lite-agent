from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from lite_agent.agent import Agent
from lite_agent.processors.stream_chunk_processor import AssistantMessageChunk
from lite_agent.runner import AgentChunk, Runner
from lite_agent.types import AgentAssistantMessage, AgentUserMessage


class DummyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(model="dummy-model", name="Dummy Agent", instructions="This is a dummy agent for testing.")

    async def completion(self, message, record_to_file=None) -> AsyncGenerator[AgentChunk, None]:  # type: ignore  # noqa: ARG002
        async def async_gen() -> AsyncGenerator[AgentChunk, None]:
            yield AssistantMessageChunk(message=AgentAssistantMessage(content="done"))

        return async_gen()


@pytest.mark.asyncio
async def test_run_until_complete():
    mock_agent = Mock()

    async def async_gen(_: object, record_to_file=None) -> AsyncGenerator[AgentChunk, None]:  # noqa: ARG001
        yield AssistantMessageChunk(message=AgentAssistantMessage(content="done"))

    mock_agent.completion = AsyncMock(side_effect=async_gen)
    runner = Runner(agent=mock_agent)
    result = await runner.run_until_complete("hello")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].type == "assistant_message"
    mock_agent.completion.assert_called_once()


@pytest.mark.asyncio
async def test_run():
    runner = Runner(agent=DummyAgent())
    gen = runner.run("hello")

    # run_stream 返回的是 async generator
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_runner_init():
    """Test Runner initialization"""
    agent = DummyAgent()
    runner = Runner(agent=agent)
    assert runner.agent == agent
    assert runner.messages == []


@pytest.mark.asyncio
async def test_runner_append_message():
    """Test Runner append_message method"""
    agent = DummyAgent()
    runner = Runner(agent=agent)

    # Test appending string message directly via append_message
    user_msg = AgentUserMessage(role="user", content="Hello")
    runner.append_message(user_msg)
    assert len(runner.messages) == 1
    assert isinstance(runner.messages[0], AgentUserMessage)
    assert runner.messages[0].role == "user"
    assert runner.messages[0].content == "Hello"

    # Test appending message object from dict
    user_msg_dict = {"role": "user", "content": "How are you?"}
    runner.append_message(user_msg_dict)
    assert len(runner.messages) == 2
    assert isinstance(runner.messages[1], AgentUserMessage)
    assert runner.messages[1].content == "How are you?"


@pytest.mark.asyncio
async def test_run_stream_with_list_input():
    """Test run_stream with list of messages as input"""
    agent = DummyAgent()
    runner = Runner(agent=agent)

    messages = [
        AgentUserMessage(role="user", content="First message"),
        AgentUserMessage(role="user", content="Second message"),
    ]

    gen = runner.run(messages)
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 1
    # Messages include the two input messages plus the assistant response
    assert len(runner.messages) == 3


@pytest.mark.asyncio
async def test_run_stream_with_record_to():
    """Test run_stream with record_to parameter"""
    agent = DummyAgent()
    runner = Runner(agent=agent)

    gen = runner.run("hello", record_to="test_record.jsonl")
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_run_stream_with_max_steps():
    """Test run_stream with custom max_steps"""
    agent = DummyAgent()
    runner = Runner(agent=agent)

    gen = runner.run("hello", max_steps=5)
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_run_continue_stream_with_invalid_last_message():
    """Test run_continue_stream when last message is not from assistant"""
    agent = DummyAgent()
    runner = Runner(agent=agent)

    # Add a user message as the last message
    user_msg_dict = {"role": "user", "content": "Hello"}
    runner.append_message(user_msg_dict)

    with pytest.raises(ValueError, match="Cannot continue running without a valid last message from the assistant"):
        async for _ in runner.run_continue_stream():
            pass


@pytest.mark.asyncio
async def test_run_continue_stream_with_empty_messages():
    """Test run_continue_stream when there are no messages"""
    agent = DummyAgent()
    runner = Runner(agent=agent)

    with pytest.raises(ValueError, match="Cannot continue running without a valid last message from the assistant"):
        async for _ in runner.run_continue_stream():
            pass


@pytest.mark.asyncio
async def test_run_continue_stream_with_tool_calls():
    """Test run_continue_stream with tool calls in last assistant message"""
    agent = DummyAgent()
    runner = Runner(agent=agent)

    # In the new format, create an assistant message and a function call message
    from lite_agent.types import AgentAssistantMessage, AgentFunctionToolCallMessage

    assistant_msg = AgentAssistantMessage(
        content="Let me call a tool",
    )

    function_call_msg = AgentFunctionToolCallMessage(
        function_call_id="test_id",
        name="test_tool",
        arguments="{}",
    )

    runner.messages.append(assistant_msg)
    runner.messages.append(function_call_msg)

    # Mock the agent.handle_tool_calls method
    from lite_agent.types import ToolCallChunk, ToolCallResultChunk

    async def mock_handle_tool_calls(tool_calls, context=None) -> AsyncGenerator[ToolCallChunk | ToolCallResultChunk, None]:  # type: ignore  # noqa: ARG001
        yield ToolCallChunk(name="test_tool", arguments="{}", id="test_id")
        yield ToolCallResultChunk(tool_call_id="test_id", name="test_tool", content="result")

    with patch.object(agent, "handle_tool_calls", side_effect=mock_handle_tool_calls):
        results = []
        async for chunk in runner.run_continue_stream():
            results.append(chunk)

        assert len(results) >= 2  # At least the tool call chunks
