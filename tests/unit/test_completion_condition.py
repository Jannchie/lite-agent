"""Test completion condition functionality."""

from lite_agent import Agent, Runner
from lite_agent.types import AgentFunctionCallOutput, AgentFunctionToolCallMessage


def test_agent_default_completion_condition():
    """Test that agents default to 'stop' completion condition."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
    )
    assert agent.completion_condition == "stop"


def test_agent_call_completion_condition():
    """Test that agents can be configured with 'call' completion condition."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="call",
    )
    assert agent.completion_condition == "call"


def test_task_done_tool_added_for_call_condition():
    """Test that task_done tool is added when completion_condition='call'."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="call",
    )

    # Check that task_done tool is registered
    assert "task_done" in agent.fc.function_registry

    # Check that task_done tool has correct metadata
    tools = agent.fc.get_tools(target="completion")
    task_done_tool = next((tool for tool in tools if tool["function"]["name"] == "task_done"), None)
    assert task_done_tool is not None
    assert task_done_tool["function"]["description"] == "Call this function when you have completed your assigned task"


def test_task_done_tool_not_added_for_stop_condition():
    """Test that task_done tool is not added when completion_condition='stop'."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="stop",
    )

    # Check that task_done tool is not registered
    assert "task_done" not in agent.fc.function_registry


def test_task_done_instructions_added():
    """Test that task_done instructions are added for call completion condition."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Original instructions",
        completion_condition="call",
    )

    messages = agent.prepare_completion_messages([])
    system_message = messages[0]

    # Check that task_done instructions are included
    assert "task_done" in system_message["content"]
    assert "When you have completed your assigned task" in system_message["content"]


def test_task_done_instructions_not_added_for_stop():
    """Test that task_done instructions are not added for stop completion condition."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Original instructions",
        completion_condition="stop",
    )

    messages = agent.prepare_completion_messages([])
    system_message = messages[0]

    # Check that task_done instructions are not included
    assert "task_done" not in system_message["content"].lower()


def test_runner_task_done_detection():
    """Test that runner can detect when task_done was called."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="call",
    )

    runner = Runner(agent)

    # Simulate adding messages including a task_done call
    runner.messages.extend([
        AgentFunctionToolCallMessage(
            type="function_call",
            function_call_id="call_123",
            name="task_done",
            arguments='{"summary": "Task completed"}',
            content="",
        ),
        AgentFunctionCallOutput(
            type="function_call_output",
            call_id="call_123",
            output="Task completed.",
        ),
    ])

    # Test that task_done was detected
    assert runner._task_done_called() is True


def test_runner_task_done_not_detected():
    """Test that runner correctly detects when task_done was not called."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="call",
    )

    runner = Runner(agent)

    # Simulate adding messages without task_done call
    runner.messages.extend([
        AgentFunctionToolCallMessage(
            type="function_call",
            function_call_id="call_123",
            name="some_other_function",
            arguments="{}",
            content="",
        ),
        AgentFunctionCallOutput(
            type="function_call_output",
            call_id="call_123",
            output="Some other result.",
        ),
    ])

    # Test that task_done was not detected
    assert runner._task_done_called() is False


async def test_task_done_function_execution():
    """Test that the task_done function can be executed."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="call",
    )

    # Call task_done function directly
    result = await agent.fc.call_function_async("task_done", '{"summary": "Test completed"}')
    assert "Task completed" in result
    assert "Test completed" in result


async def test_task_done_function_execution_no_summary():
    """Test that the task_done function works without summary."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="call",
    )

    # Call task_done function without summary
    result = await agent.fc.call_function_async("task_done", "{}")
    assert result == "Task completed."
