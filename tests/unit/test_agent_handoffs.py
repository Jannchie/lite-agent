"""Unit tests for agent handoff functionality in the runner."""

import json

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import ToolCall, ToolCallFunction


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
        enum_values = transfer_tool["parameters"]["properties"]["name"]["enum"]
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
        assert "Transferring to agent: SalesAgent" in result

        # Test invalid transfer
        result = await main_agent.fc.call_function_async(
            "transfer_to_agent",
            '{"name": "InvalidAgent"}',
        )
        assert "not found" in result
        assert "SalesAgent" in result

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
        await runner._handle_agent_transfer(transfer_call, [])

        # Verify agent was switched
        assert runner.agent.name == "SalesAgent"

        # Verify function call output was added
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert output_msg.type == "function_call_output"
        assert output_msg.call_id == "test_transfer_001"
        assert "SalesAgent" in output_msg.output

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
        await runner._handle_agent_transfer(invalid_transfer_call, [])

        # Agent should remain unchanged
        assert runner.agent.name == "MainAgent"

        # Error result should be added to messages
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert output_msg.type == "function_call_output"
        assert "not found" in output_msg.output

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
        async for chunk in runner._handle_tool_calls(tool_calls, ["tool_call_result"]):
            chunks.append(chunk)

        # Verify agent was transferred
        assert runner.agent.name == "SalesAgent"

        # Verify transfer result added to messages
        transfer_outputs = [msg for msg in runner.messages if hasattr(msg, "type") and msg.type == "function_call_output"]
        assert len(transfer_outputs) == 1
        assert "SalesAgent" in transfer_outputs[0].output

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
        await runner._handle_agent_transfer(transfer_call, [])

        # Agent should remain unchanged
        assert runner.agent.name == "MainAgent"

        # Error result should be added to messages
        assert len(runner.messages) == 1
        output_msg = runner.messages[0]
        assert output_msg.type == "function_call_output"
        assert "no handoffs configured" in output_msg.output
