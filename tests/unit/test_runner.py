from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from lite_agent.agent import Agent
from lite_agent.processors.completion_event_processor import AssistantMessageEvent
from lite_agent.runner import AgentChunk, Runner
from lite_agent.types import AgentAssistantMessage, AgentUserMessage, UserTextContent


class DummyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(model="dummy-model", name="Dummy Agent", instructions="This is a dummy agent for testing.")

    async def completion(self, _message, record_to_file=None) -> AsyncGenerator[AgentChunk, None]:  # type: ignore
        async def async_gen() -> AsyncGenerator[AgentChunk, None]:
            yield AssistantMessageEvent(message=AgentAssistantMessage(content="done"))

        return async_gen()

    async def responses(self, _message, record_to_file=None) -> AsyncGenerator[AgentChunk, None]:  # type: ignore
        async def async_gen() -> AsyncGenerator[AgentChunk, None]:
            yield AssistantMessageEvent(message=AgentAssistantMessage(content="done"))

        return async_gen()


@pytest.mark.asyncio
async def test_run_until_complete():
    mock_agent = Mock()

    async def async_gen(_: object, record_to_file=None) -> AsyncGenerator[AgentChunk, None]:
        yield AssistantMessageEvent(message=AgentAssistantMessage(content="done"))

    mock_agent.completion = AsyncMock(side_effect=async_gen)
    runner = Runner(agent=mock_agent, api="completion")
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
    from lite_agent.types import NewUserMessage

    agent = DummyAgent()
    runner = Runner(agent=agent)

    # Test appending string message directly via append_message
    user_msg = AgentUserMessage(role="user", content="Hello")
    runner.append_message(user_msg)
    assert len(runner.messages) == 1
    # Now expects NewUserMessage since append_message converts legacy to new format
    assert isinstance(runner.messages[0], NewUserMessage)
    assert runner.messages[0].role == "user"
    assert len(runner.messages[0].content) == 1
    assert isinstance(runner.messages[0].content[0], UserTextContent)
    assert runner.messages[0].content[0].text == "Hello"

    # Test appending message object from dict
    user_msg_dict = {"role": "user", "content": "How are you?"}
    runner.append_message(user_msg_dict)
    assert len(runner.messages) == 2
    assert isinstance(runner.messages[1], NewUserMessage)
    assert isinstance(runner.messages[1].content[0], UserTextContent)
    assert runner.messages[1].content[0].text == "How are you?"


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
    from lite_agent.types import AssistantTextContent, AssistantToolCall, NewAssistantMessage

    agent = DummyAgent()
    runner = Runner(agent=agent)

    # Create an assistant message with a tool call using the new format
    assistant_msg = NewAssistantMessage(
        content=[
            AssistantTextContent(text="Let me call a tool"),
            AssistantToolCall(
                call_id="test_id",
                name="test_tool",
                arguments="{}",
            ),
        ],
    )

    runner.messages.append(assistant_msg)

    # Mock the agent.handle_tool_calls method
    from lite_agent.types import FunctionCallEvent, FunctionCallOutputEvent

    async def mock_handle_tool_calls(tool_calls, context=None) -> AsyncGenerator[FunctionCallEvent | FunctionCallOutputEvent, None]:  # type: ignore
        yield FunctionCallEvent(name="test_tool", arguments="{}", call_id="test_id")
        yield FunctionCallOutputEvent(tool_call_id="test_id", name="test_tool", content="result")

    with patch.object(agent, "handle_tool_calls", side_effect=mock_handle_tool_calls):
        results = []
        async for chunk in runner.run_continue_stream():
            results.append(chunk)

        assert len(results) >= 2  # At least the tool call chunks
