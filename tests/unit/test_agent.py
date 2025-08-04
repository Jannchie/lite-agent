from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lite_agent.agent import Agent
from lite_agent.types import AgentUserMessage, RunnerMessage, ToolCall, ToolCallFunction, UserTextContent


@pytest.mark.asyncio
async def test_prepare_completion_messages():
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)
    messages: list[RunnerMessage] = [AgentUserMessage(content=[UserTextContent(text="hi")])]
    result = agent.prepare_completion_messages(messages)
    assert result[0]["role"] == "system"
    assert "TestBot" in result[0]["content"]
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "hi"


@pytest.mark.asyncio
async def test_stream_async_success():
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)
    agent.fc.get_tools = MagicMock(return_value=[{"name": "tool1"}])
    fake_resp = MagicMock()
    with patch("lite_agent.client.litellm.acompletion", new=AsyncMock(return_value=fake_resp)), patch("lite_agent.agent.CustomStreamWrapper", new=lambda _: True):
        # Patch Agent模块作用域下的litellm_completion_stream_handler
        from collections.abc import AsyncGenerator
        from typing import Any

        async def fake_async_gen(*_args, **_kwargs) -> AsyncGenerator[Any, None]:  # type: ignore
            yield "GENERATOR"

        with patch("lite_agent.agent.litellm_completion_stream_handler", new=fake_async_gen), patch("lite_agent.agent.isinstance", new=lambda _obj, _typ: True):
            result = await agent.completion([AgentUserMessage(content=[UserTextContent(text="hi")])])
            assert hasattr(result, "__aiter__")
            items = []
            async for item in result:
                items.append(item)
            assert items == ["GENERATOR"]


@pytest.mark.asyncio
async def test_stream_async_typeerror():
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)
    agent.fc.get_tools = MagicMock(return_value=[{"name": "tool1"}])
    not_a_stream = object()

    class DummyWrapper:
        pass

    with (
        patch("lite_agent.client.litellm.acompletion", new=AsyncMock(return_value=not_a_stream)),
        patch("lite_agent.agent.CustomStreamWrapper", DummyWrapper),
        pytest.raises(TypeError, match="Response is not a CustomStreamWrapper"),
    ):
        await agent.completion([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_list_require_confirm_tools_empty_input():
    """Test list_require_confirm_tools with None input"""
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)
    result = await agent.list_require_confirm_tools(None)
    assert result == []


@pytest.mark.asyncio
async def test_list_require_confirm_tools_not_found():
    """Test list_require_confirm_tools when tool function is not found in registry"""

    def dummy_tool() -> str:
        return "dummy"

    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=[dummy_tool])

    # Mock a tool call with a function that doesn't exist in registry
    tool_call = ToolCall(
        id="test_id",
        function=ToolCallFunction(name="nonexistent_tool", arguments="{}"),
        type="function",
        index=0,
    )

    with patch("lite_agent.agent.logger") as mock_logger:
        result = await agent.list_require_confirm_tools([tool_call])
        assert result == []
        mock_logger.warning.assert_called_once_with("Tool function %s not found in registry", "nonexistent_tool")


@pytest.mark.asyncio
async def test_list_require_confirm_tools_with_confirmation():
    """Test list_require_confirm_tools when tool requires confirmation"""
    from funcall.decorators import tool

    @tool(require_confirmation=True)
    def confirm_tool() -> str:
        return "dummy_result"

    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=[confirm_tool])

    tool_call = ToolCall(
        id="test_id",
        function=ToolCallFunction(name="confirm_tool", arguments="{}"),
        type="function",
        index=0,
    )

    result = await agent.list_require_confirm_tools([tool_call])
    assert len(result) == 1
    assert result[0] == tool_call


@pytest.mark.asyncio
async def test_handle_tool_calls_empty_input():
    """Test handle_tool_calls with None input"""
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)

    result = agent.handle_tool_calls(None)
    # The function returns early if tool_calls is None/empty
    # We need to check if it's an async generator that produces no results
    items = []
    async for item in result:
        items.append(item)
    assert items == []


@pytest.mark.asyncio
async def test_handle_tool_calls_function_not_found():
    """Test handle_tool_calls when tool function is not found in registry"""

    def dummy_tool() -> str:
        return "dummy"

    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=[dummy_tool])

    tool_call = ToolCall(
        id="test_id",
        function=ToolCallFunction(name="nonexistent_tool", arguments="{}"),
        type="function",
        index=0,
    )

    with patch("lite_agent.agent.logger") as mock_logger:
        items = []
        async for item in agent.handle_tool_calls([tool_call]):
            items.append(item)

        # Current behavior: still yields items even for non-existent tools
        # because the continue only applies to the first loop, not the second
        assert len(items) == 2  # FunctionCallEvent + FunctionCallOutputEvent with error
        assert items[0].type == "function_call"
        assert items[0].name == "nonexistent_tool"
        assert items[1].type == "function_call_output"
        # The call will fail during execution, not during the registry check
        mock_logger.warning.assert_called_once_with("Tool function %s not found in registry", "nonexistent_tool")


@pytest.mark.asyncio
async def test_handle_tool_calls_exception():
    """Test handle_tool_calls when tool execution raises an exception"""

    def failing_tool() -> str:
        msg = "Tool execution failed"
        raise RuntimeError(msg)

    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=[failing_tool])

    tool_call = ToolCall(
        id="test_id",
        function=ToolCallFunction(name="failing_tool", arguments="{}"),
        type="function",
        index=0,
    )

    with patch("lite_agent.agent.logger") as mock_logger:
        items = []
        async for item in agent.handle_tool_calls([tool_call]):
            items.append(item)

        # Should yield 2 items: FunctionCallEvent and FunctionCallOutputEvent with error
        assert len(items) == 2
        assert items[0].type == "function_call"
        assert items[0].name == "failing_tool"
        assert items[1].type == "function_call_output"
        assert items[1].tool_call_id == "test_id"
        assert "Tool execution failed" in items[1].content

        mock_logger.exception.assert_called_once_with("Tool call %s failed", "test_id")


@pytest.mark.asyncio
async def test_handle_tool_calls_success():
    """Test handle_tool_calls with successful tool execution"""

    def working_tool() -> str:
        return "tool_result"

    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=[working_tool])

    tool_call = ToolCall(
        id="test_id",
        function=ToolCallFunction(name="working_tool", arguments="{}"),
        type="function",
        index=0,
    )

    items = []
    async for item in agent.handle_tool_calls([tool_call]):
        items.append(item)

    # Should yield 2 items: FunctionCallEvent and FunctionCallOutputEvent with success
    assert len(items) == 2
    assert items[0].type == "function_call"
    assert items[0].name == "working_tool"
    assert items[1].type == "function_call_output"
    assert items[1].tool_call_id == "test_id"
    assert items[1].content == "tool_result"
