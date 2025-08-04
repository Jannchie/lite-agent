"""
Additional tests for message_transfers.py to improve coverage
"""

from lite_agent.message_transfers import (
    _process_dict_message,
    _process_function_message,
    _process_message_to_xml,
    consolidate_history_transfer,
)
from lite_agent.types import (
    AgentAssistantMessage,
    AgentUserMessage,
    AssistantTextContent,
    NewAssistantMessage,
    NewUserMessage,
    UserTextContent,
)


class TestMessageTransfersAdditional:
    """Additional tests for message transfers functionality"""

    def test_process_dict_message_function_call(self):
        """Test _process_dict_message with function call"""
        message = {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"city": "Tokyo"}',
        }
        result = _process_dict_message(message)
        assert len(result) == 1
        assert "function_call" in result[0]
        assert "get_weather" in result[0]
        assert "Tokyo" in result[0]

    def test_process_dict_message_function_output(self):
        """Test _process_dict_message with function call output"""
        message = {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "The weather is sunny",
        }
        result = _process_dict_message(message)
        assert len(result) == 1
        assert "function_result" in result[0]
        assert "call_123" in result[0]
        assert "sunny" in result[0]

    def test_process_dict_message_regular_roles(self):
        """Test _process_dict_message with regular role messages"""
        # Test user message
        user_message = {
            "role": "user",
            "content": "Hello, how are you?",
        }
        result = _process_dict_message(user_message)
        assert len(result) == 1
        assert "role='user'" in result[0]
        assert "Hello, how are you?" in result[0]

        # Test assistant message
        assistant_message = {
            "role": "assistant",
            "content": "I'm doing well, thank you!",
        }
        result = _process_dict_message(assistant_message)
        assert len(result) == 1
        assert "role='assistant'" in result[0]
        assert "I'm doing well" in result[0]

        # Test system message
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant",
        }
        result = _process_dict_message(system_message)
        assert len(result) == 1
        assert "role='system'" in result[0]
        assert "helpful assistant" in result[0]

    def test_process_dict_message_with_defaults(self):
        """Test _process_dict_message with missing fields"""
        # Test empty message - should return empty list
        message = {}
        result = _process_dict_message(message)
        assert len(result) == 0

        # Test message with only role
        message = {"role": "user"}
        result = _process_dict_message(message)
        assert len(result) == 1
        assert "role='user'" in result[0]
        assert "message" in result[0]

        # Test message with content but no role
        message = {"content": "test"}
        result = _process_dict_message(message)
        assert len(result) == 0  # No role, so no message element created    def test_process_function_message_with_attributes(self):
        """Test _process_function_message with object having attributes"""

        class MockMessage:
            def __init__(self):
                self.type = "function_call"
                self.name = "test_function"
                self.arguments = '{"param": "value"}'

        message = MockMessage()
        result = _process_function_message(message)
        assert len(result) == 1
        assert "test_function" in result[0]
        assert "param" in result[0]

    def test_process_function_message_output_with_attributes(self):
        """Test _process_function_message with function output"""

        class MockMessage:
            def __init__(self):
                self.type = "function_call_output"
                self.call_id = "call_456"
                self.output = "Function executed successfully"

        message = MockMessage()
        result = _process_function_message(message)
        assert len(result) == 1
        assert "call_456" in result[0]
        assert "successfully" in result[0]

    def test_process_function_message_unknown_type(self):
        """Test _process_function_message with unknown type"""

        class MockMessage:
            def __init__(self):
                self.type = "unknown_type"

        message = MockMessage()
        result = _process_function_message(message)
        assert len(result) == 0

    def test_process_function_message_with_defaults(self):
        """Test _process_function_message with missing attributes"""

        class MockMessage:
            def __init__(self):
                self.type = "function_call"
                # Missing name and arguments

        message = MockMessage()
        result = _process_function_message(message)
        assert len(result) == 1
        assert "name='unknown'" in result[0]
        assert "arguments=''" in result[0]

    def test_process_message_to_xml_with_agent_messages(self):
        """Test _process_message_to_xml with agent message objects"""
        # Test with AgentUserMessage containing string content
        user_msg = AgentUserMessage(role="user", content="Hello world")
        result = _process_message_to_xml(user_msg)
        assert len(result) == 1
        assert "role='user'" in result[0]
        assert "Hello world" in result[0]

        # Test with AgentAssistantMessage containing string content
        assistant_msg = AgentAssistantMessage(role="assistant", content="Hi there")
        result = _process_message_to_xml(assistant_msg)
        assert len(result) == 1
        assert "role='assistant'" in result[0]
        assert "Hi there" in result[0]

    def test_process_message_to_xml_with_new_format_messages(self):
        """Test _process_message_to_xml with new format messages"""
        # Test with NewUserMessage
        user_msg = NewUserMessage(
            content=[
                UserTextContent(text="Hello"),
                UserTextContent(text="World"),
            ],
        )
        result = _process_message_to_xml(user_msg)
        assert len(result) >= 1
        # Should contain both text contents
        xml_str = " ".join(result)
        assert "Hello" in xml_str
        assert "World" in xml_str

        # Test with NewAssistantMessage
        assistant_msg = NewAssistantMessage(
            content=[
                AssistantTextContent(text="Response text"),
            ],
        )
        result = _process_message_to_xml(assistant_msg)
        assert len(result) >= 1
        xml_str = " ".join(result)
        assert "Response text" in xml_str

    def test_process_message_to_xml_with_function_type_attribute(self):
        """Test _process_message_to_xml with message having type attribute"""

        class MockFunctionMessage:
            def __init__(self):
                self.type = "function_call"
                self.name = "mock_function"
                self.arguments = "{}"

        message = MockFunctionMessage()
        result = _process_message_to_xml(message)
        assert len(result) == 1
        assert "mock_function" in result[0]

    def test_consolidate_history_transfer_with_mixed_types(self):
        """Test consolidate_history_transfer with mixed message types"""
        messages = [
            # Dict message
            {"role": "user", "content": "User message"},
            # Agent message
            AgentAssistantMessage(role="assistant", content="Assistant response"),
            # New format message
            NewUserMessage(content=[UserTextContent(text="New format user message")]),
            # Function call
            {
                "type": "function_call",
                "name": "test_function",
                "arguments": '{"test": "value"}',
            },
            # Function output
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Function result",
            },
        ]

        result = consolidate_history_transfer(messages)
        assert len(result) == 1

        # Handle both dict and Pydantic model formats
        message = result[0]
        if isinstance(message, dict):
            assert message["role"] == "user"
            content_str = message["content"]
        else:
            assert message.role == "user"
            # Content is a list for new message format
            content_str = ""
            if hasattr(message, "content"):
                if isinstance(message.content, list):
                    # Extract text from content items
                    for item in message.content:
                        if hasattr(item, "text") and hasattr(item, "type") and item.type == "text":
                            content_str += item.text
                elif isinstance(message.content, str):
                    content_str = message.content

        assert "conversation_history" in content_str
        assert "User message" in content_str
        assert "Assistant response" in content_str
        assert "New format user message" in content_str
        assert "test_function" in content_str
        assert "Function result" in content_str
        assert "接下来该做什么?" in content_str

    def test_consolidate_history_transfer_empty_list(self):
        """Test consolidate_history_transfer with empty message list"""
        result = consolidate_history_transfer([])
        assert result == []

    def test_consolidate_history_transfer_with_complex_content(self):
        """Test consolidate_history_transfer with complex content structures"""
        # Test message with complex content that needs XML escaping
        messages = [
            {"role": "user", "content": "Message with <special> & characters"},
            {
                "type": "function_call",
                "name": "complex_function",
                "arguments": '{"data": "<xml>content</xml>"}',
            },
        ]

        result = consolidate_history_transfer(messages)
        assert len(result) == 1

        # Handle both dict and Pydantic model formats
        message = result[0]
        if isinstance(message, dict):
            content = message["content"]
        elif isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, list):
            content = ""
            for item in message.content:
                if hasattr(item, "text") and hasattr(item, "type") and item.type == "text":
                    content += item.text
        else:
            content = ""

        assert "conversation_history" in content
        assert "special" in content
        assert "complex_function" in content
        assert "xml" in content

    def test_process_message_to_xml_edge_cases(self):
        """Test _process_message_to_xml with edge cases"""
        # Test with empty content list
        empty_user_msg = NewUserMessage(content=[])
        result = _process_message_to_xml(empty_user_msg)
        # Should still produce some output even with empty content
        assert isinstance(result, list)

        # Test with message having no recognizable pattern
        class UnknownMessage:
            pass

        unknown_msg = UnknownMessage()
        result = _process_message_to_xml(unknown_msg)
        assert isinstance(result, list)

    def test_process_message_to_xml_with_dict_messages(self):
        """Test _process_message_to_xml with dictionary messages"""
        # Test standard dict message
        dict_msg = {"role": "user", "content": "Dict message"}
        result = _process_message_to_xml(dict_msg)
        assert len(result) == 1
        assert "Dict message" in result[0]

        # Test function call dict
        function_dict = {
            "type": "function_call",
            "name": "test_func",
            "arguments": "{}",
        }
        result = _process_message_to_xml(function_dict)
        assert len(result) == 1
        assert "test_func" in result[0]
