"""Unit tests for the set_chat_history functionality - updated for NewMessage-only."""

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import AssistantToolCall, AssistantToolCallResult, NewAssistantMessage, NewUserMessage, UserTextContent


def get_temperature(city: str) -> str:
    """Mock function to get temperature."""
    return f"The temperature in {city} is 25°C."


def get_weather(city: str) -> str:
    """Mock function to get weather."""
    return f"The weather in {city} is sunny."


class TestSetChatHistoryNew:
    """Test cases for the set_chat_history method with NewMessage format."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create parent agent
        self.parent = Agent(
            model="gpt-4.1-nano",
            name="ParentAgent",
            instructions="You are the main assistant.",
            tools=[get_temperature],
        )

        # Create weather agent
        self.weather_agent = Agent(
            model="gpt-4.1-nano",
            name="WeatherAgent",
            instructions="You are a weather specialist.",
            tools=[get_weather],
        )
        self.weather_agent.parent = self.parent

        # Add the weather agent as handoff to parent
        self.parent.add_handoff(self.weather_agent)

        # Create runner
        self.runner = Runner(self.parent)

    def test_set_chat_history_basic(self):
        """Test basic chat history setting without transfers."""
        user_message = NewUserMessage(content=[UserTextContent(text="Hello")])
        assistant_message = NewAssistantMessage(content=[])

        messages = [user_message, assistant_message]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2
        assert self.runner.agent.name == "ParentAgent"  # Should remain as parent

    def test_set_chat_history_with_transfer_to_agent(self):
        """Test chat history setting with transfer_to_agent function call."""
        # Create user message
        user_message = NewUserMessage(content=[UserTextContent(text="I need weather info")])

        # Create assistant message with transfer
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

        messages = [user_message, assistant_message]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2
        assert self.runner.agent.name == "WeatherAgent"  # Should transfer to WeatherAgent

    def test_set_chat_history_with_transfer_to_parent(self):
        """Test chat history setting with transfer_to_parent function call."""
        # Start with weather agent
        weather_runner = Runner(self.weather_agent)

        # Create messages including transfer back to parent
        user_message = NewUserMessage(content=[UserTextContent(text="Thanks for the weather info")])
        assistant_message = NewAssistantMessage(
            content=[
                AssistantToolCall(
                    call_id="call_1",
                    name="transfer_to_parent",
                    arguments="{}",
                ),
                AssistantToolCallResult(
                    call_id="call_1",
                    output="Transferring back to parent",
                ),
            ],
        )

        messages = [user_message, assistant_message]

        weather_runner.set_chat_history(messages, root_agent=self.weather_agent)

        assert len(weather_runner.messages) == 2
        assert weather_runner.agent.name == "ParentAgent"  # Should transfer back to parent

    def test_set_chat_history_with_agent_objects(self):
        """Test chat history setting with tool calls in new message format."""
        # Create user message
        user_message = NewUserMessage(content=[UserTextContent(text="Hello")])

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

        messages = [user_message, assistant_message]

        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2
        assert self.runner.agent.name == "WeatherAgent"

    def test_set_chat_history_clears_previous_messages(self):
        """Test that set_chat_history clears previous messages."""
        # Add an initial message
        initial_message = NewUserMessage(content=[UserTextContent(text="Initial message")])
        self.runner.append_message(initial_message)
        assert len(self.runner.messages) == 1

        # Set new chat history
        new_user_message = NewUserMessage(content=[UserTextContent(text="Hello")])
        new_assistant_message = NewAssistantMessage(content=[])
        messages = [new_user_message, new_assistant_message]

        self.runner.set_chat_history(messages)

        # Should have new messages, not the initial one
        assert len(self.runner.messages) == 2
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "Hello"

    def test_set_chat_history_without_root_agent(self):
        """Test set_chat_history without specifying root_agent."""
        user_message = NewUserMessage(content=[UserTextContent(text="Hello")])
        assistant_message = NewAssistantMessage(content=[])
        messages = [user_message, assistant_message]

        # Should use self.agent as default
        original_agent = self.runner.agent
        self.runner.set_chat_history(messages)

        assert len(self.runner.messages) == 2
        assert self.runner.agent == original_agent

    def test_dict_format_converted_in_set_chat_history(self):
        """Test that dict format messages are converted in set_chat_history."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Dict messages should be converted to NewMessage format
        self.runner.set_chat_history(messages, root_agent=self.parent)

        assert len(self.runner.messages) == 2
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert isinstance(self.runner.messages[1], NewAssistantMessage)
        from lite_agent.types import UserTextContent

        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "Hello"

    def test_find_agent_by_name(self):
        """Test the _find_agent_by_name method."""
        # Test finding the weather agent
        found_agent = self.runner._find_agent_by_name(self.parent, "WeatherAgent")
        assert found_agent is not None
        assert found_agent.name == "WeatherAgent"

        # Test finding non-existent agent
        not_found = self.runner._find_agent_by_name(self.parent, "NonExistentAgent")
        assert not_found is None
