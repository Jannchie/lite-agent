import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import AssistantTextContent, AssistantToolCall, NewAssistantMessage


class TestCancelPendingTools:
    """Test cancellation of pending tool calls when new user input is provided."""

    def test_cancel_pending_tool_calls_method(self):
        """Test the _cancel_pending_tool_calls method directly."""

        def dummy_tool(param: str) -> str:
            return f"Result for {param}"

        agent = Agent(
            model="gpt-4.1-nano",
            name="TestAgent",
            instructions="Test agent",
            tools=[dummy_tool],
        )

        runner = Runner(agent)

        # Add an assistant message with pending tool calls
        assistant_message = NewAssistantMessage(
            content=[
                AssistantTextContent(text="I'll help you."),
                AssistantToolCall(
                    call_id="call_1",
                    name="dummy_tool",
                    arguments='{"param": "test1"}',
                ),
                AssistantToolCall(
                    call_id="call_2",
                    name="dummy_tool",
                    arguments='{"param": "test2"}',
                ),
            ],
        )

        runner.messages.append(assistant_message)

        # Verify we have pending calls
        pending_calls = runner._find_pending_tool_calls()
        assert len(pending_calls) == 2

        # Cancel pending calls and get events
        cancellation_events = runner._cancel_pending_tool_calls()

        # Verify cancellation events were generated
        assert len(cancellation_events) == 2
        for i, event in enumerate(cancellation_events):
            assert event.type == "function_call_output"
            assert event.tool_call_id in ["call_1", "call_2"]
            assert event.name == "dummy_tool"
            assert event.content == "Operation cancelled by user - new input provided"
            assert event.execution_time_ms == 0

        # Verify no more pending calls
        pending_calls_after = runner._find_pending_tool_calls()
        assert len(pending_calls_after) == 0

        # Verify cancellation results were added
        assert len(runner.messages) == 1
        assert isinstance(runner.messages[0], NewAssistantMessage)

        assistant_msg = runner.messages[0]
        assert len(assistant_msg.content) == 5  # text + 2 tool_calls + 2 tool_call_results

        # Check that cancellation results were added
        cancellation_results = [item for item in assistant_msg.content if item.type == "tool_call_result"]
        assert len(cancellation_results) == 2

        for result in cancellation_results:
            assert result.output == "Operation cancelled by user - new input provided"
            assert result.execution_time_ms == 0

    def test_cancel_pending_tool_calls_no_pending(self):
        """Test that _cancel_pending_tool_calls does nothing when no pending calls exist."""
        agent = Agent(
            model="gpt-4",
            name="TestAgent",
            instructions="Test agent",
        )

        runner = Runner(agent)

        # No messages at all
        assert len(runner.messages) == 0

        # Should not raise any errors
        runner._cancel_pending_tool_calls()

        # Still no messages
        assert len(runner.messages) == 0

    def test_cancel_pending_tool_calls_with_completed_calls(self):
        """Test that only pending calls are cancelled, completed ones are left alone."""

        def dummy_tool(param: str) -> str:
            return f"Result for {param}"

        agent = Agent(
            model="gpt-4",
            name="TestAgent",
            instructions="Test agent",
            tools=[dummy_tool],
        )

        runner = Runner(agent)

        # Add an assistant message with both pending and completed tool calls
        assistant_message = NewAssistantMessage(
            content=[
                AssistantTextContent(text="I'll help you."),
                AssistantToolCall(
                    call_id="call_completed",
                    name="dummy_tool",
                    arguments='{"param": "completed"}',
                ),
                AssistantToolCall(
                    call_id="call_pending",
                    name="dummy_tool",
                    arguments='{"param": "pending"}',
                ),
            ],
        )

        # Add a result for the completed call, but not for the pending one
        runner.messages.append(assistant_message)
        runner._add_tool_call_result("call_completed", "Completed result", 100)

        # Verify we have 1 pending call
        pending_calls = runner._find_pending_tool_calls()
        assert len(pending_calls) == 1
        assert pending_calls[0].call_id == "call_pending"

        # Cancel pending calls
        runner._cancel_pending_tool_calls()

        # Verify no more pending calls
        pending_calls_after = runner._find_pending_tool_calls()
        assert len(pending_calls_after) == 0

        # Verify the completed call result is still there and unchanged
        assistant_msg = runner.messages[0]
        results = [item for item in assistant_msg.content if item.type == "tool_call_result"]
        assert len(results) == 2  # One completed, one cancelled

        # Find the completed result
        completed_result = next((r for r in results if r.call_id == "call_completed"), None)
        assert completed_result is not None
        assert completed_result.output == "Completed result"
        assert completed_result.execution_time_ms == 100

        # Find the cancelled result
        cancelled_result = next((r for r in results if r.call_id == "call_pending"), None)
        assert cancelled_result is not None
        assert cancelled_result.output == "Operation cancelled by user - new input provided"
        assert cancelled_result.execution_time_ms == 0

    def test_cancel_pending_tool_calls_no_events_when_empty(self):
        """Test that no events are generated when there are no pending calls."""
        agent = Agent(
            model="gpt-4",
            name="TestAgent",
            instructions="Test agent",
        )

        runner = Runner(agent)

        # No pending calls
        cancellation_events = runner._cancel_pending_tool_calls()

        # Should return empty list
        assert cancellation_events == []

    @pytest.mark.asyncio
    async def test_run_method_yields_cancellation_events(self):
        """Test that run() method yields cancellation events when user provides new input."""

        def dummy_tool(param: str) -> str:
            return f"Result for {param}"

        agent = Agent(
            model="gpt-4",
            name="TestAgent",
            instructions="Test agent",
            tools=[dummy_tool],
        )

        runner = Runner(agent)

        # Add an assistant message with pending tool calls
        assistant_message = NewAssistantMessage(
            content=[
                AssistantTextContent(text="I'm working on it."),
                AssistantToolCall(
                    call_id="pending_1",
                    name="dummy_tool",
                    arguments='{"param": "test"}',
                ),
            ],
        )

        runner.messages.append(assistant_message)

        # Verify we have pending calls
        pending_calls = runner._find_pending_tool_calls()
        assert len(pending_calls) == 1

        # Provide new user input - this should generate cancellation events
        # We'll collect the first few chunks before the LLM call fails
        chunks = []
        try:
            async for chunk in runner.run("What's 2+2?", includes=["function_call_output"]):
                chunks.append(chunk)
                # Break early to avoid actual LLM call
                if len(chunks) >= 1:
                    break
        except Exception:
            # Expected - no actual LLM configured
            pass

        # Should have gotten cancellation event
        assert len(chunks) >= 1
        cancellation_chunk = chunks[0]
        assert cancellation_chunk.type == "function_call_output"
        assert cancellation_chunk.tool_call_id == "pending_1"
        assert cancellation_chunk.name == "dummy_tool"
        assert "Operation cancelled by user" in cancellation_chunk.content
