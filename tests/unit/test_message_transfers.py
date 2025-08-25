"""Test the predefined message transfer functions."""

from lite_agent.message_transfers import consolidate_history_transfer
from lite_agent.types import NewUserMessage, NewAssistantMessage, UserTextContent, AssistantTextContent, AssistantToolCall, AssistantToolCallResult


def test_consolidate_history_transfer_basic():
    """Test basic functionality of consolidate_history_transfer."""
    messages = [
        NewUserMessage(content=[UserTextContent(text="Hello")]),
        NewAssistantMessage(content=[AssistantTextContent(text="Hi there!")]),
    ]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    assert isinstance(result[0], NewUserMessage)
    assert result[0].role == "user"
    first_content = result[0].content[0]
    assert isinstance(first_content, UserTextContent)
    content_text = first_content.text
    assert "以下是目前发生的所有交互:" in content_text
    assert "<conversation_history>" in content_text
    assert "<message role='user'>Hello</message>" in content_text
    assert "<message role='assistant'>Hi there!</message>" in content_text
    assert "接下来该做什么?" in content_text


def test_consolidate_history_transfer_with_function_calls():
    """Test consolidate_history_transfer with function calls."""
    messages = [
        NewUserMessage(content=[UserTextContent(text="Check the weather")]),
        NewAssistantMessage(content=[
            AssistantToolCall(call_id="call_123", name="get_weather", arguments='{"city": "Tokyo"}'),
            AssistantToolCallResult(call_id="call_123", output="Sunny, 22°C"),
            AssistantTextContent(text="The weather in Tokyo is sunny and 22°C."),
        ]),
    ]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    assert isinstance(result[0], NewUserMessage)
    first_content = result[0].content[0]
    assert isinstance(first_content, UserTextContent)
    content = first_content.text
    assert "<function_call name='get_weather' arguments='{\"city\": \"Tokyo\"}' />" in content
    assert "<function_result call_id='call_123'>Sunny, 22°C</function_result>" in content


def test_consolidate_history_transfer_empty():
    """Test consolidate_history_transfer with empty messages."""
    result = consolidate_history_transfer([])
    assert result == []


def test_consolidate_history_transfer_single_message():
    """Test consolidate_history_transfer with a single message."""
    messages = [NewUserMessage(content=[UserTextContent(text="Test message")])]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    assert isinstance(result[0], NewUserMessage)
    first_content = result[0].content[0]
    assert isinstance(first_content, UserTextContent)
    content = first_content.text
    assert "<message role='user'>Test message</message>" in content


def test_consolidate_history_transfer_mixed_types():
    """Test consolidate_history_transfer with mixed message types."""
    messages = [
        NewUserMessage(content=[UserTextContent(text="Pydantic message")]),
        NewAssistantMessage(content=[
            AssistantTextContent(text="Dict message"),
            AssistantToolCall(call_id="test_123", name="test_func", arguments="{}"),
        ]),
    ]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    assert isinstance(result[0], NewUserMessage)
    first_content = result[0].content[0]
    assert isinstance(first_content, UserTextContent)
    content = first_content.text
    assert "<message role='user'>Pydantic message</message>" in content
    assert "<message role='assistant'>Dict message</message>" in content
    assert "<function_call name='test_func' arguments='{}' />" in content
