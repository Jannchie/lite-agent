"""Test completion condition functionality."""

from lite_agent import Agent


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


def test_wait_for_user_tool_added_for_call_condition():
    """Test that wait_for_user tool is added when completion_condition='call'."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="call",
    )

    # Check that wait_for_user tool is registered
    assert "wait_for_user" in agent.fc.function_registry

    # Check that wait_for_user tool has correct metadata
    tools = agent.fc.get_tools(target="completion")
    wait_for_user_tool = next((tool for tool in tools if tool["function"]["name"] == "wait_for_user"), None)
    assert wait_for_user_tool is not None


def test_wait_for_user_tool_not_added_for_stop_condition():
    """Test that wait_for_user tool is not added when completion_condition='stop'."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Test instructions",
        completion_condition="stop",
    )

    # Check that wait_for_user tool is not registered
    assert "wait_for_user" not in agent.fc.function_registry


def test_wait_for_user_instructions_added():
    """Test that wait_for_user instructions are added for call completion condition."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Original instructions",
        completion_condition="call",
    )

    messages = agent.prepare_completion_messages([])
    system_message = messages[0]

    # Check that wait_for_user instructions are included
    assert "wait_for_user" in system_message["content"]
    assert "When you have completed your assigned task" in system_message["content"]


def test_wait_for_user_instructions_not_added_for_stop():
    """Test that wait_for_user instructions are not added for stop completion condition."""
    agent = Agent(
        name="TestAgent",
        model="gpt-4o-mini",
        instructions="Original instructions",
        completion_condition="stop",
    )

    messages = agent.prepare_completion_messages([])
    system_message = messages[0]

    # Check that wait_for_user instructions are not included
    assert "wait_for_user" not in system_message["content"].lower()
