"""Tests for message builder utility."""

from lite_agent.types import (
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    UserImageContent,
    UserTextContent,
)
from lite_agent.utils.message_builder import MessageBuilder


def test_build_user_message_from_dict_string_content():
    """Test building user message from dict with string content."""
    message_dict = {
        "role": "user",
        "content": "Hello, world!",
    }

    result = MessageBuilder.build_user_message_from_dict(message_dict)

    assert result.role == "user"
    assert len(result.content) == 1
    assert isinstance(result.content[0], UserTextContent)
    assert result.content[0].text == "Hello, world!"


def test_build_user_message_from_dict_with_meta():
    """Test building user message from dict with meta information."""
    from datetime import datetime, timezone

    test_time = datetime.now(timezone.utc)
    message_dict = {
        "role": "user",
        "content": "Test message",
        "meta": {"sent_at": test_time},
    }

    result = MessageBuilder.build_user_message_from_dict(message_dict)

    assert result.role == "user"
    assert result.meta.sent_at == test_time


def test_build_user_message_from_dict_list_content():
    """Test building user message from dict with list content."""
    message_dict = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ],
    }

    result = MessageBuilder.build_user_message_from_dict(message_dict)

    assert result.role == "user"
    assert len(result.content) == 2
    assert all(isinstance(item, UserTextContent) for item in result.content)
    assert result.content[0].text == "Hello"
    assert result.content[1].text == "World"


def test_build_user_message_from_dict_non_string_content():
    """Test building user message from dict with non-string content."""
    message_dict = {
        "role": "user",
        "content": 123,
    }

    result = MessageBuilder.build_user_message_from_dict(message_dict)

    assert result.role == "user"
    assert len(result.content) == 1
    assert isinstance(result.content[0], UserTextContent)
    assert result.content[0].text == "123"


def test_build_user_content_from_dict_text():
    """Test building user content from text dict."""
    item = {"type": "text", "text": "Test text"}
    result = MessageBuilder._build_user_content_from_dict(item)

    assert isinstance(result, UserTextContent)
    assert result.text == "Test text"


def test_build_user_content_from_dict_input_text():
    """Test building user content from input_text dict."""
    item = {"type": "input_text", "text": "Input text"}
    result = MessageBuilder._build_user_content_from_dict(item)

    assert isinstance(result, UserTextContent)
    assert result.text == "Input text"


def test_build_user_content_from_dict_image_url():
    """Test building user content from image_url dict."""
    item = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.jpg"},
    }
    result = MessageBuilder._build_user_content_from_dict(item)

    assert isinstance(result, UserImageContent)
    assert result.image_url == "https://example.com/image.jpg"


def test_build_user_content_from_dict_image_url_string():
    """Test building user content from image_url dict with string value."""
    item = {
        "type": "image_url",
        "image_url": "https://example.com/image.jpg",
    }
    result = MessageBuilder._build_user_content_from_dict(item)

    assert isinstance(result, UserImageContent)
    assert result.image_url == "https://example.com/image.jpg"


def test_build_user_content_from_dict_input_image():
    """Test building user content from input_image dict."""
    item = {
        "type": "input_image",
        "image_url": "https://example.com/image.jpg",
        "file_id": "file123",
        "detail": "high",
    }
    result = MessageBuilder._build_user_content_from_dict(item)

    assert isinstance(result, UserImageContent)
    assert result.image_url == "https://example.com/image.jpg"
    assert result.file_id == "file123"
    assert result.detail == "high"


def test_build_user_content_from_dict_fallback():
    """Test building user content from dict with unknown type."""
    item = {"type": "unknown", "text": "fallback text"}
    result = MessageBuilder._build_user_content_from_dict(item)

    assert isinstance(result, UserTextContent)
    assert result.text == "fallback text"


def test_build_user_content_from_object_input_text():
    """Test building user content from input_text object."""

    class MockItem:
        type = "input_text"
        text = "Object text"

    result = MessageBuilder._build_user_content_from_object(MockItem())

    assert isinstance(result, UserTextContent)
    assert result.text == "Object text"


def test_build_user_content_from_object_input_image():
    """Test building user content from input_image object."""

    class MockItem:
        type = "input_image"
        image_url = "https://example.com/obj.jpg"
        file_id = "obj123"
        detail = "low"

    result = MessageBuilder._build_user_content_from_object(MockItem())

    assert isinstance(result, UserImageContent)
    assert result.image_url == "https://example.com/obj.jpg"
    assert result.file_id == "obj123"
    assert result.detail == "low"


def test_build_user_content_from_object_fallback():
    """Test building user content from object with unknown type."""

    class MockItem:
        type = "unknown"
        value = "test"

    result = MessageBuilder._build_user_content_from_object(MockItem())

    assert isinstance(result, UserTextContent)


def test_build_user_content_items_mixed():
    """Test building user content items from mixed list."""
    content_list = [
        {"type": "text", "text": "Dict text"},
        "Plain string",
        123,
    ]

    # Add a mock object to the list
    class MockItem:
        type = "input_text"
        text = "Object text"

    content_list.append(MockItem())

    result = MessageBuilder._build_user_content_items(content_list)

    assert len(result) == 4
    assert all(isinstance(item, UserTextContent) for item in result)
    assert result[0].text == "Dict text"
    assert result[1].text == "Plain string"
    assert result[2].text == "123"
    assert result[3].text == "Object text"


def test_build_system_message_from_dict():
    """Test building system message from dict."""
    from datetime import datetime, timezone

    test_time = datetime.now(timezone.utc)
    message_dict = {
        "role": "system",
        "content": "You are a helpful assistant.",
        "meta": {"sent_at": test_time},
    }

    result = MessageBuilder.build_system_message_from_dict(message_dict)

    assert result.role == "system"
    assert result.content == "You are a helpful assistant."
    assert result.meta.sent_at == test_time


def test_build_system_message_from_dict_no_meta():
    """Test building system message from dict without meta."""
    message_dict = {
        "role": "system",
        "content": "System prompt",
    }

    result = MessageBuilder.build_system_message_from_dict(message_dict)

    assert result.role == "system"
    assert result.content == "System prompt"
    assert result.meta is not None  # Should have default meta


def test_build_assistant_message_from_dict_string_content():
    """Test building assistant message from dict with string content."""
    message_dict = {
        "role": "assistant",
        "content": "Hello, how can I help?",
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 1
    assert isinstance(result.content[0], AssistantTextContent)
    assert result.content[0].text == "Hello, how can I help?"


def test_build_assistant_message_from_dict_list_content():
    """Test building assistant message from dict with list content."""
    message_dict = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll help you"},
            {"type": "tool_call", "call_id": "call123", "name": "search", "arguments": '{"q": "test"}'},
            {"type": "tool_call_result", "call_id": "call123", "output": "Found results", "execution_time_ms": 500},
        ],
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 3

    assert isinstance(result.content[0], AssistantTextContent)
    assert result.content[0].text == "I'll help you"

    assert isinstance(result.content[1], AssistantToolCall)
    assert result.content[1].call_id == "call123"
    assert result.content[1].name == "search"
    assert result.content[1].arguments == '{"q": "test"}'

    assert isinstance(result.content[2], AssistantToolCallResult)
    assert result.content[2].call_id == "call123"
    assert result.content[2].output == "Found results"
    assert result.content[2].execution_time_ms == 500


def test_build_assistant_message_from_dict_with_tool_calls():
    """Test building assistant message from dict with legacy tool_calls format."""
    message_dict = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call456",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
            },
        ],
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 1
    assert isinstance(result.content[0], AssistantToolCall)
    assert result.content[0].call_id == "call456"
    assert result.content[0].name == "get_weather"
    # Arguments should be parsed as dict
    assert result.content[0].arguments == {"city": "Tokyo"}


def test_build_assistant_message_from_dict_tool_calls_invalid_json():
    """Test building assistant message with invalid JSON in tool_calls arguments."""
    message_dict = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call789",
                "function": {
                    "name": "invalid_func",
                    "arguments": "invalid json {",
                },
            },
        ],
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 1
    assert isinstance(result.content[0], AssistantToolCall)
    assert result.content[0].call_id == "call789"
    assert result.content[0].name == "invalid_func"
    # Should keep original string when JSON parsing fails
    assert result.content[0].arguments == "invalid json {"


def test_build_assistant_message_from_dict_tool_calls_dict_arguments():
    """Test building assistant message with dict arguments in tool_calls."""
    message_dict = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call999",
                "function": {
                    "name": "dict_func",
                    "arguments": {"param": "value"},
                },
            },
        ],
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 1
    assert isinstance(result.content[0], AssistantToolCall)
    assert result.content[0].arguments == {"param": "value"}


def test_build_assistant_message_from_dict_list_content_fallback():
    """Test building assistant message with unknown list content item."""
    message_dict = {
        "role": "assistant",
        "content": [
            {"type": "unknown", "data": "test"},  # Now handled as unknown dict type
            "plain string",
            123,
        ],
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 3  # All items are processed
    assert all(isinstance(item, AssistantTextContent) for item in result.content)
    assert result.content[0].text == "{'type': 'unknown', 'data': 'test'}"
    assert result.content[1].text == "plain string"
    assert result.content[2].text == "123"


def test_build_assistant_message_from_dict_non_string_non_list_content():
    """Test building assistant message with non-string, non-list content."""
    message_dict = {
        "role": "assistant",
        "content": {"key": "value"},
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 1
    assert isinstance(result.content[0], AssistantTextContent)
    assert result.content[0].text == "{'key': 'value'}"


def test_build_assistant_message_from_dict_empty_content():
    """Test building assistant message with empty content."""
    message_dict = {
        "role": "assistant",
        "content": "",
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 0


def test_build_assistant_message_from_dict_no_content():
    """Test building assistant message with no content key."""
    message_dict = {
        "role": "assistant",
    }

    result = MessageBuilder.build_assistant_message_from_dict(message_dict)

    assert result.role == "assistant"
    assert len(result.content) == 0
