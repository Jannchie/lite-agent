"""Tests for the new structured message types."""

from lite_agent.types.messages import (
    AssistantMessageContent,
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    MessageMeta,
    MessageUsage,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    UserImageContent,
    UserMessageContent,
    UserTextContent,
    assistant_message_to_llm_dict,
    user_message_to_llm_dict,
)


def test_user_message_creation():
    """Test creating a new user message with text content."""
    content: list[UserMessageContent] = [UserTextContent(text="Hello, world!")]
    message = NewUserMessage(content=content)

    assert message.role == "user"
    assert len(message.content) == 1
    assert message.content[0].type == "text"
    assert message.content[0].text == "Hello, world!"
    assert isinstance(message.meta, MessageMeta)


def test_user_message_with_image():
    """Test creating a user message with image content."""
    content = [
        UserTextContent(text="Look at this image:"),
        UserImageContent(image_url="https://example.com/image.jpg"),
    ]
    message = NewUserMessage(content=content)

    assert len(message.content) == 2
    assert message.content[0].type == "text"
    assert message.content[1].type == "image"
    assert message.content[1].image_url == "https://example.com/image.jpg"


def test_system_message_creation():
    """Test creating a system message."""
    message = NewSystemMessage(content="You are a helpful assistant.")

    assert message.role == "system"
    assert message.content == "You are a helpful assistant."
    assert isinstance(message.meta, MessageMeta)


def test_assistant_message_with_text():
    """Test creating an assistant message with text content."""
    content: list[AssistantMessageContent] = [AssistantTextContent(text="I can help you with that.")]
    message = NewAssistantMessage(content=content)

    assert message.role == "assistant"
    assert len(message.content) == 1
    assert message.content[0].type == "text"
    assert message.content[0].text == "I can help you with that."
    assert isinstance(message.meta, AssistantMessageMeta)


def test_assistant_message_with_tool_calls():
    """Test creating an assistant message with tool calls and results."""
    content = [
        AssistantTextContent(text="I'll check the weather for you."),
        AssistantToolCall(
            call_id="call_123",
            name="get_weather",
            arguments={"location": "New York"},
        ),
        AssistantToolCallResult(
            call_id="call_123",
            output="Temperature: 22°C, Sunny",
            execution_time_ms=150,
        ),
        AssistantTextContent(text="The weather in New York is 22°C and sunny."),
    ]

    message = NewAssistantMessage(content=content)

    assert len(message.content) == 4
    assert message.content[0].type == "text"
    assert message.content[1].type == "tool_call"
    assert message.content[2].type == "tool_call_result"
    assert message.content[3].type == "text"


def test_assistant_message_meta_with_usage():
    """Test assistant message metadata with usage statistics."""
    usage = MessageUsage(input_tokens=50, output_tokens=25, total_tokens=75)
    meta = AssistantMessageMeta(
        model="gpt-4",
        usage=usage,
        total_time_ms=1500,
        latency_ms=200,
    )

    content: list[AssistantMessageContent] = [AssistantTextContent(text="Response")]
    message = NewAssistantMessage(content=content, meta=meta)

    assert message.meta.model == "gpt-4"
    assert message.meta.usage is not None
    assert message.meta.usage.input_tokens == 50
    assert message.meta.usage.output_tokens == 25
    assert message.meta.usage.total_tokens == 75
    assert message.meta.total_time_ms == 1500
    assert message.meta.latency_ms == 200


def test_to_llm_dict_user_message():
    """Test converting user message to LLM dict format."""
    # Single text content
    content: list[UserMessageContent] = [UserTextContent(text="Hello")]
    message = NewUserMessage(content=content)
    llm_dict = user_message_to_llm_dict(message)

    assert llm_dict["role"] == "user"
    assert llm_dict["content"] == "Hello"

    # Multiple content items
    content = [
        UserTextContent(text="Hello"),
        UserImageContent(image_url="https://example.com/image.jpg"),
    ]
    message = NewUserMessage(content=content)
    llm_dict = user_message_to_llm_dict(message)

    assert llm_dict["role"] == "user"
    assert isinstance(llm_dict["content"], list)
    assert len(llm_dict["content"]) == 2


def test_to_llm_dict_assistant_message():
    """Test converting assistant message to LLM dict format."""
    content = [
        AssistantTextContent(text="I can help"),
        AssistantToolCall(
            call_id="call_123",
            name="get_weather",
            arguments={"location": "NYC"},
        ),
    ]
    message = NewAssistantMessage(content=content)
    llm_dict = assistant_message_to_llm_dict(message)

    assert llm_dict["role"] == "assistant"
    assert llm_dict["content"] == "I can help"
    assert "tool_calls" in llm_dict
    assert len(llm_dict["tool_calls"]) == 1
    assert llm_dict["tool_calls"][0]["id"] == "call_123"
