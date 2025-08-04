"""Unit tests for agent handoff functionality in the runner."""

import json
from typing import Any, cast

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import NewAssistantMessage, ToolCall, ToolCallFunction


class TestAgentHandoffs:
    """Test cases for agent handoff functionality."""

    def test_agent_handoff_tool_registration(self):
        """Test that handoff agents get transfer tools registered."""
        sales_agent = Agent(
            model="gpt-4",
            name="SalesAgent",
            instructions="Sales specialist",
        )

        support_agent = Agent(
            model="gpt-4",
            name="SupportAgent",
            instructions="Support specialist",
        )

        main_agent = Agent(
            model="gpt-4",
            name="MainAgent",
            instructions="Main agent",
            handoffs=[sales_agent, support_agent],
        )

        # Check that transfer tool was created
        tools = main_agent.fc.get_tools(target="completion")
        transfer_tools = [tool for tool in tools if tool["function"]["name"] == "transfer_to_agent"]

        assert len(transfer_tools) == 1
        transfer_tool = transfer_tools[0]["function"]
        assert transfer_tool["name"] == "transfer_to_agent"
        assert "parameters" in transfer_tool

        # Check enum contains handoff agent names
        transfer_tool_params = cast("dict[str, Any]", transfer_tool["parameters"])
        properties = cast("dict[str, Any]", transfer_tool_params["properties"])
        name_prop = cast("dict[str, Any]", properties["name"])
        enum_values = cast("list[str]", name_prop["enum"])
        assert "SalesAgent" in enum_values
        assert "SupportAgent" in enum_values

    @pytest.mark.asyncio
    async def test_transfer_to_agent_function(self):
        """Test the transfer_to_agent function directly."""
        sales_agent = Agent(
            model="gpt-4",
            name="SalesAgent",
            instructions="Sales specialist",
        )

        main_agent = Agent(
            model="gpt-4",
            name="MainAgent",
            instructions="Main agent",
            handoffs=[sales_agent],
        )

        # Test valid transfer
        result = await main_agent.fc.call_function_async(
            "transfer_to_agent",
            '{"name": "SalesAgent"}',
        )
        result_str = cast("str", result)
        assert "Transferring to agent: SalesAgent" in result_str

        # Test invalid transfer
        result = await main_agent.fc.call_function_async(
            "transfer_to_agent",
            '{"name": "InvalidAgent"}',
        )
        result_str = cast("str", result)
        assert "not found" in result_str
        assert "SalesAgent" in result_str

    @pytest.mark.asyncio
    async def test_runner_agent_transfer(self):
        """Test that runner correctly handles agent transfers."""
        sales_agent = Agent(
            model="gpt-4",
            name="SalesAgent",
            instructions="Sales specialist",
        )

        main_agent = Agent(
            model="gpt-4",
            name="MainAgent",
            instructions="Main agent",
            handoffs=[sales_agent],
        )

        runner = Runner(main_agent)

        # Verify initial state
        assert runner.agent.name == "MainAgent"
        assert len(runner.messages) == 0

        # Create transfer tool call
        transfer_call = ToolCall(
            id="test_transfer_001",
            type="function",
            function=ToolCallFunction(
                name="transfer_to_agent",
                arguments=json.dumps({"name": "SalesAgent"}),
            ),
            index=0,
        )

        # Handle the transfer
        await runner._handle_agent_transfer(transfer_call)

        # Verify agent was switched
        assert runner.agent.name == "SalesAgent"

        # Verify function call output was added (now as NewAssistantMessage with tool result)
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 1
        tool_result = output_msg.content[0]
        assert tool_result.type == "tool_call_result"
        assert tool_result.call_id == "test_transfer_001"
        assert "SalesAgent" in tool_result.output

    @pytest.mark.asyncio
    async def test_runner_invalid_agent_transfer(self):
        """Test runner handling of invalid agent transfers."""
        sales_agent = Agent(
            model="gpt-4",
            name="SalesAgent",
            instructions="Sales specialist",
        )

        main_agent = Agent(
            model="gpt-4",
            name="MainAgent",
            instructions="Main agent",
            handoffs=[sales_agent],
        )

        runner = Runner(main_agent)

        # Create invalid transfer call
        invalid_transfer_call = ToolCall(
            id="test_invalid_001",
            type="function",
            function=ToolCallFunction(
                name="transfer_to_agent",
                arguments=json.dumps({"name": "NonExistentAgent"}),
            ),
            index=0,
        )

        # Handle the invalid transfer
        await runner._handle_agent_transfer(invalid_transfer_call)

        # Agent should remain unchanged
        assert runner.agent.name == "MainAgent"

        # Error result should be added to messages (now as NewAssistantMessage with tool result)
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 1
        tool_result = output_msg.content[0]
        assert tool_result.type == "tool_call_result"
        assert "not found" in tool_result.output

    @pytest.mark.asyncio
    async def test_runner_handle_tool_calls_with_transfer(self):
        """Test that _handle_tool_calls processes transfers correctly."""
        sales_agent = Agent(
            model="gpt-4",
            name="SalesAgent",
            instructions="Sales specialist",
        )

        main_agent = Agent(
            model="gpt-4",
            name="MainAgent",
            instructions="Main agent",
            handoffs=[sales_agent],
        )

        runner = Runner(main_agent)

        # Create mixed tool calls (transfer + regular)
        transfer_call = ToolCall(
            id="transfer_001",
            type="function",
            function=ToolCallFunction(
                name="transfer_to_agent",
                arguments=json.dumps({"name": "SalesAgent"}),
            ),
            index=0,
        )

        tool_calls = [transfer_call]

        # Process tool calls
        chunks = []
        async for chunk in runner._handle_tool_calls(tool_calls, ["function_call_output"]):
            chunks.append(chunk)

        # Verify agent was transferred
        assert runner.agent.name == "SalesAgent"

        # Verify transfer result added to messages (now as NewAssistantMessage with tool result)
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 1
        tool_result = output_msg.content[0]
        assert tool_result.type == "tool_call_result"
        assert "SalesAgent" in tool_result.output

    @pytest.mark.asyncio
    async def test_runner_no_handoffs_configured(self):
        """Test transfer handling when no handoffs are configured."""
        main_agent = Agent(
            model="gpt-4",
            name="MainAgent",
            instructions="Main agent",
            # No handoffs configured
        )

        runner = Runner(main_agent)

        # This should not crash but should log error
        transfer_call = ToolCall(
            id="invalid_001",
            type="function",
            function=ToolCallFunction(
                name="transfer_to_agent",
                arguments=json.dumps({"name": "SomeAgent"}),
            ),
            index=0,
        )

        # Should handle gracefully
        await runner._handle_agent_transfer(transfer_call)

        # Agent should remain unchanged
        assert runner.agent.name == "MainAgent"

        # Error result should be added to messages (now as NewAssistantMessage with tool result)
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 1
        tool_result = output_msg.content[0]
        assert tool_result.type == "tool_call_result"
        assert "no handoffs configured" in tool_result.output

    @pytest.mark.asyncio
    async def test_handle_parent_transfer_success(self):
        """Test successful transfer to parent agent."""
        # Create parent and child agents
        parent_agent = Agent(
            model="gpt-4",
            name="ParentAgent",
            instructions="Parent agent",
        )

        child_agent = Agent(
            model="gpt-4",
            name="ChildAgent",
            instructions="Child agent",
        )

        # Set parent manually
        child_agent.parent = parent_agent

        runner = Runner(agent=child_agent)

        # Create a transfer_to_parent tool call
        transfer_call = ToolCall(
            type="function",
            id="call_123",
            function=ToolCallFunction(
                name="transfer_to_parent",
                arguments="{}",
            ),
            index=0,
        )

        # Should transfer successfully
        await runner._handle_parent_transfer(transfer_call)

        # Agent should have switched to parent
        assert runner.agent.name == "ParentAgent"
        assert runner.agent == parent_agent

        # Function call output should be added to messages (now as NewAssistantMessage with tool result)
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 1
        tool_result = output_msg.content[0]
        assert tool_result.type == "tool_call_result"
        assert tool_result.call_id == "call_123"

    @pytest.mark.asyncio
    async def test_handle_parent_transfer_no_parent(self):
        """Test transfer to parent when agent has no parent."""
        agent = Agent(
            model="gpt-4",
            name="MainAgent",
            instructions="Main agent without parent",
        )

        runner = Runner(agent=agent)

        # Create a transfer_to_parent tool call
        transfer_call = ToolCall(
            type="function",
            id="call_456",
            function=ToolCallFunction(
                name="transfer_to_parent",
                arguments="{}",
            ),
            index=0,
        )

        # Should handle gracefully
        await runner._handle_parent_transfer(transfer_call)

        # Agent should remain unchanged
        assert runner.agent.name == "MainAgent"

        # Error result should be added to messages (now as NewAssistantMessage with tool result)
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 1
        tool_result = output_msg.content[0]
        assert tool_result.type == "tool_call_result"
        assert tool_result.call_id == "call_456"
        assert "no parent to transfer back to" in tool_result.output

    @pytest.mark.asyncio
    async def test_handle_tool_calls_with_parent_transfer(self):
        """Test _handle_tool_calls with transfer_to_parent tool call."""
        # Create parent and child agents
        parent_agent = Agent(
            model="gpt-4",
            name="ParentAgent",
            instructions="Parent agent",
        )

        child_agent = Agent(
            model="gpt-4",
            name="ChildAgent",
            instructions="Child agent",
        )

        # Set parent manually
        child_agent.parent = parent_agent

        runner = Runner(agent=child_agent)

        # Create a transfer_to_parent tool call
        transfer_call = ToolCall(
            type="function",
            id="call_789",
            function=ToolCallFunction(
                name="transfer_to_parent",
                arguments="{}",
            ),
            index=0,
        )

        # Call _handle_tool_calls
        chunks = []
        async for chunk in runner._handle_tool_calls([transfer_call], ["function_call_output"]):
            chunks.append(chunk)

        # Should return immediately after transfer (no chunks yielded)
        assert len(chunks) == 0

        # Agent should have switched to parent
        assert runner.agent.name == "ParentAgent"
        assert runner.agent == parent_agent

        # Function call output should be added to messages (now as NewAssistantMessage with tool result)
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 1
        tool_result = output_msg.content[0]
        assert tool_result.type == "tool_call_result"
        assert tool_result.call_id == "call_789"

    @pytest.mark.asyncio
    async def test_handle_tool_calls_with_multiple_parent_transfers(self):
        """Test _handle_tool_calls with multiple transfer_to_parent calls."""
        # Create parent and child agents
        parent_agent = Agent(
            model="gpt-4",
            name="ParentAgent",
            instructions="Parent agent",
        )

        child_agent = Agent(
            model="gpt-4",
            name="ChildAgent",
            instructions="Child agent",
        )

        # Set parent manually
        child_agent.parent = parent_agent

        runner = Runner(agent=child_agent)

        # Create multiple transfer_to_parent tool calls
        transfer_call_1 = ToolCall(
            type="function",
            id="call_111",
            function=ToolCallFunction(
                name="transfer_to_parent",
                arguments="{}",
            ),
            index=0,
        )

        transfer_call_2 = ToolCall(
            type="function",
            id="call_222",
            function=ToolCallFunction(
                name="transfer_to_parent",
                arguments="{}",
            ),
            index=1,
        )

        # Call _handle_tool_calls
        chunks = []
        async for chunk in runner._handle_tool_calls([transfer_call_1, transfer_call_2], ["function_call_output"]):
            chunks.append(chunk)

        # Should return immediately after transfer (no chunks yielded)
        assert len(chunks) == 0

        # Agent should have switched to parent
        assert runner.agent.name == "ParentAgent"
        assert runner.agent == parent_agent

        # Should have 1 assistant message with 2 function call outputs
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert isinstance(output_msg, NewAssistantMessage)
        assert len(output_msg.content) == 2

        # First call should execute
        tool_result_1 = output_msg.content[0]
        assert tool_result_1.type == "tool_call_result"
        assert tool_result_1.call_id == "call_111"

        # Second call should be skipped
        tool_result_2 = output_msg.content[1]
        assert tool_result_2.type == "tool_call_result"
        assert tool_result_2.call_id == "call_222"
        assert "Transfer already executed" in tool_result_2.output
