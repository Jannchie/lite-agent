"""Unit tests for the set_chat_history functionality in Runner class."""

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import AssistantToolCall, AssistantToolCallResult, NewAssistantMessage


def get_temperature(city: str) -> str:
    """Mock function to get temperature."""
    return f"The temperature in {city} is 25°C."


def get_weather(city: str) -> str:
    """Mock function to get weather."""
    return f"The weather in {city} is sunny."


class TestSetChatHistory:
    """Test cases for the set_chat_history method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parent = Agent(
            model="gpt-4.1",
            name="ParentAgent",
            instructions="You are a helpful parent agent.",
        )

        self.weather_agent = Agent(
            model="gpt-4.1",
            name="WeatherAgent",
            instructions="You are a weather specialist agent.",
            tools=[get_weather],
        )

        self.temperature_agent = Agent(
            model="gpt-4.1",
            name="TemperatureAgent",
            instructions="You are a temperature specialist agent.",
            tools=[get_temperature],
        )

        self.parent.add_handoff(self.weather_agent)
        self.parent.add_handoff(self.temperature_agent)

        self.runner = Runner(self.parent)

    def test_set_chat_history_basic(self):
        """Test basic chat history setting without transfers."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2
        assert self.runner.agent.name == "ParentAgent"

    def test_set_chat_history_with_transfer_to_agent(self):
        """Test chat history setting with transfer_to_agent function call."""
        messages = [
            {"role": "user", "content": "I need weather info"},
            {"role": "assistant", "content": "Let me transfer you to our weather specialist."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "transfer_to_agent",
                "arguments": '{"name": "WeatherAgent"}',
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Transferring to agent: WeatherAgent",
            },
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2  # In new format: user message + aggregated assistant message
        assert self.runner.agent.name == "WeatherAgent"

    def test_set_chat_history_with_transfer_to_parent(self):
        """Test chat history setting with transfer_to_parent function call."""
        # Start with child agent
        messages = [
            {"role": "user", "content": "I need weather info"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "transfer_to_agent",
                "arguments": '{"name": "WeatherAgent"}',
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Transferring to agent: WeatherAgent",
            },
            {"role": "assistant", "content": "Weather info provided"},
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "transfer_to_parent",
                "arguments": "{}",
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "Transferring back to parent",
            },
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 3  # In new format: user message + 2 separate assistant messages
        assert self.runner.agent.name == "ParentAgent"

    def test_set_chat_history_complex_transfers(self):
        """Test complex agent transfers: parent -> child -> parent -> another child."""
        messages = [
            {"role": "user", "content": "I need weather and temperature info"},
            # Transfer to weather agent
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "transfer_to_agent",
                "arguments": '{"name": "WeatherAgent"}',
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Transferring to agent: WeatherAgent",
            },
            {"role": "assistant", "content": "Weather info provided"},
            # Transfer back to parent
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "transfer_to_parent",
                "arguments": "{}",
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "Transferring back to parent",
            },
            {"role": "assistant", "content": "Now let me get temperature info"},
            # Transfer to temperature agent
            {
                "type": "function_call",
                "call_id": "call_3",
                "name": "transfer_to_agent",
                "arguments": '{"name": "TemperatureAgent"}',
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_3",
                "output": "Transferring to agent: TemperatureAgent",
            },
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 4  # In new format: user message + 3 separate assistant messages
        assert self.runner.agent.name == "TemperatureAgent"

    def test_set_chat_history_with_agent_objects(self):
        """Test chat history setting with tool calls in new message format."""
        # Create assistant message with tool call and result using new format
        assistant_message = NewAssistantMessage(
            content=[
                AssistantToolCall(
                    call_id="call_1",
                    name="transfer_to_agent",
                    arguments='{"name": "WeatherAgent"}',
                ),
                AssistantToolCallResult(
                    call_id="call_1",
                    output="Transferring to agent: WeatherAgent",
                ),
            ],
        )

        messages = [
            {"role": "user", "content": "Hello"},
            assistant_message,
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2
        assert self.runner.agent.name == "WeatherAgent"

    def test_set_chat_history_invalid_agent_name(self):
        """Test chat history with invalid agent name (should stay on current agent)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "transfer_to_agent",
                "arguments": '{"name": "NonExistentAgent"}',
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Agent not found",
            },
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2  # In new format: user message + aggregated assistant message
        assert self.runner.agent.name == "ParentAgent"  # Should stay on parent

    def test_set_chat_history_malformed_arguments(self):
        """Test chat history with malformed transfer arguments."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "transfer_to_agent",
                "arguments": "invalid_json",
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Transfer failed",
            },
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2  # In new format: user message + aggregated assistant message
        assert self.runner.agent.name == "ParentAgent"  # Should stay on parent

    def test_set_chat_history_transfer_to_parent_without_parent(self):
        """Test transfer_to_parent when agent has no parent."""
        # Create a standalone agent without parent
        standalone_agent = Agent(
            model="gpt-4.1",
            name="StandaloneAgent",
            instructions="I am standalone",
        )
        runner = Runner(standalone_agent)

        messages = [
            {"role": "user", "content": "Hello"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "transfer_to_parent",
                "arguments": "{}",
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "No parent to transfer to",
            },
        ]

        runner.set_chat_history(messages, root_agent=standalone_agent)

        assert len(runner.messages) == 2  # In new format: user message + aggregated assistant message
        assert runner.agent.name == "StandaloneAgent"  # Should stay on standalone

    def test_set_chat_history_clears_previous_messages(self):
        """Test that set_chat_history clears previous messages."""
        # Add some initial messages
        self.runner.append_message({"role": "user", "content": "Initial message"})
        assert len(self.runner.messages) == 1

        # Set new chat history
        messages = [
            {"role": "user", "content": "New message"},
            {"role": "assistant", "content": "New response"},
        ]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2
        assert hasattr(self.runner.messages[0], "content")
        assert self.runner.messages[0].content[0].text == "New message"  # type: ignore
        assert hasattr(self.runner.messages[1], "content")
        assert self.runner.messages[1].content[0].text == "New response"  # type: ignore

    def test_set_chat_history_without_root_agent(self):
        """Test set_chat_history without specifying root_agent."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Should use self.agent as default
        original_agent = self.runner.agent
        self.runner.set_chat_history(messages)

        assert len(self.runner.messages) == 2
        assert self.runner.agent == original_agent

    def test_append_message_with_various_types(self):
        """Test that append_message works with various message types."""
        # Test dict message
        dict_msg = {"role": "user", "content": "Hello"}
        self.runner.append_message(dict_msg)
        assert len(self.runner.messages) == 1

        # Test new format assistant message with tool call and result
        assistant_msg = NewAssistantMessage(
            content=[
                AssistantToolCall(
                    call_id="call_1",
                    name="test_function",
                    arguments='{"test": "value"}',
                ),
                AssistantToolCallResult(
                    call_id="call_1",
                    output="Test output",
                ),
            ],
        )
        self.runner.append_message(assistant_msg)
        assert len(self.runner.messages) == 2

    def test_find_agent_by_name(self):
        """Test the _find_agent_by_name helper method."""
        # Test finding direct handoff
        found_agent = self.runner._find_agent_by_name(self.parent, "WeatherAgent")
        assert found_agent is not None
        assert found_agent.name == "WeatherAgent"

        # Test finding another direct handoff
        found_agent = self.runner._find_agent_by_name(self.parent, "TemperatureAgent")
        assert found_agent is not None
        assert found_agent.name == "TemperatureAgent"

        # Test not finding non-existent agent
        found_agent = self.runner._find_agent_by_name(self.parent, "NonExistentAgent")
        assert found_agent is None

        # Test finding from child agent (should look up to parent's handoffs)
        found_agent = self.runner._find_agent_by_name(self.weather_agent, "TemperatureAgent")
        assert found_agent is not None
        assert found_agent.name == "TemperatureAgent"
