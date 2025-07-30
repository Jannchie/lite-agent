"""Tests for the new structured message types."""

from datetime import datetime, timezone

from lite_agent.types import (
    AgentAssistantMessage,
    AgentFunctionCallOutput,
    AgentFunctionToolCallMessage,
    AgentUserMessage,
    BasicMessageMeta,
    LLMResponseMeta,
)
from lite_agent.types.messages import (
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
    UserTextContent,
    convert_legacy_to_new,
    convert_new_to_legacy,
)


def test_user_message_creation():
    """Test creating a new user message with text content."""
    content = [UserTextContent(text="Hello, world!")]
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
    content = [AssistantTextContent(text="I can help you with that.")]
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

    content = [AssistantTextContent(text="Response")]
    message = NewAssistantMessage(content=content, meta=meta)

    assert message.meta.model == "gpt-4"
    assert message.meta.usage.input_tokens == 50
    assert message.meta.usage.output_tokens == 25
    assert message.meta.usage.total_tokens == 75
    assert message.meta.total_time_ms == 1500
    assert message.meta.latency_ms == 200


def test_to_llm_dict_user_message():
    """Test converting user message to LLM dict format."""
    # Single text content
    content = [UserTextContent(text="Hello")]
    message = NewUserMessage(content=content)
    llm_dict = message.to_llm_dict()

    assert llm_dict["role"] == "user"
    assert llm_dict["content"] == "Hello"

    # Multiple content items
    content = [
        UserTextContent(text="Hello"),
        UserImageContent(image_url="https://example.com/image.jpg"),
    ]
    message = NewUserMessage(content=content)
    llm_dict = message.to_llm_dict()

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
    llm_dict = message.to_llm_dict()

    assert llm_dict["role"] == "assistant"
    assert llm_dict["content"] == "I can help"
    assert "tool_calls" in llm_dict
    assert len(llm_dict["tool_calls"]) == 1
    assert llm_dict["tool_calls"][0]["id"] == "call_123"


def test_convert_legacy_user_message_to_new():
    """Test converting legacy user message to new format."""
    legacy_message = AgentUserMessage(
        content="Hello, world!",
        meta=BasicMessageMeta(sent_at=datetime.now(timezone.utc)),
    )

    new_messages = convert_legacy_to_new([legacy_message])

    assert len(new_messages) == 1
    assert isinstance(new_messages[0], NewUserMessage)
    assert len(new_messages[0].content) == 1
    assert new_messages[0].content[0].text == "Hello, world!"


def test_convert_legacy_assistant_message_with_tools():
    """Test converting legacy assistant message with tool calls to new format."""
    now = datetime.now(timezone.utc)

    legacy_messages = [
        AgentAssistantMessage(
            content="I'll check the weather.",
            meta=LLMResponseMeta(
                sent_at=now,
                input_tokens=20,
                output_tokens=10,
                latency_ms=150,
            ),
        ),
        AgentFunctionToolCallMessage(
            call_id="call_123",
            name="get_weather",
            arguments='{"location": "NYC"}',
            meta=BasicMessageMeta(sent_at=now),
        ),
        AgentFunctionCallOutput(
            call_id="call_123",
            output="22°C, Sunny",
            meta=BasicMessageMeta(sent_at=now, execution_time_ms=100),
        ),
    ]

    new_messages = convert_legacy_to_new(legacy_messages)

    assert len(new_messages) == 1
    assert isinstance(new_messages[0], NewAssistantMessage)

    # Check content structure
    content = new_messages[0].content
    assert len(content) == 3
    assert content[0].type == "text"
    assert content[1].type == "tool_call"
    assert content[2].type == "tool_call_result"

    # Check metadata
    assert new_messages[0].meta.usage.input_tokens == 20
    assert new_messages[0].meta.usage.output_tokens == 10
    assert new_messages[0].meta.latency_ms == 150


def test_convert_new_to_legacy():
    """Test converting new format back to legacy format."""
    content = [
        AssistantTextContent(text="I'll help"),
        AssistantToolCall(
            call_id="call_123",
            name="get_weather",
            arguments={"location": "NYC"},
        ),
        AssistantToolCallResult(
            call_id="call_123",
            output="22°C, Sunny",
            execution_time_ms=100,
        ),
    ]

    usage = MessageUsage(input_tokens=20, output_tokens=10, total_tokens=30)
    meta = AssistantMessageMeta(
        usage=usage,
        latency_ms=150,
        total_time_ms=500,
    )

    new_message = NewAssistantMessage(content=content, meta=meta)
    legacy_messages = convert_new_to_legacy([new_message])

    assert len(legacy_messages) == 3
    assert isinstance(legacy_messages[0], AgentAssistantMessage)
    assert isinstance(legacy_messages[1], AgentFunctionToolCallMessage)
    assert isinstance(legacy_messages[2], AgentFunctionCallOutput)

    # Check content preservation
    assert legacy_messages[0].content == "I'll help"
    assert legacy_messages[1].call_id == "call_123"
    assert legacy_messages[2].output == "22°C, Sunny"

    # Check metadata preservation
    assert legacy_messages[0].meta.input_tokens == 20
    assert legacy_messages[0].meta.output_tokens == 10
    assert legacy_messages[0].meta.latency_ms == 150


def test_roundtrip_conversion():
    """Test that converting legacy -> new -> legacy preserves data."""
    now = datetime.now(timezone.utc)

    original_messages = [
        AgentUserMessage(content="Hello", meta=BasicMessageMeta(sent_at=now)),
        AgentAssistantMessage(
            content="Hi there!",
            meta=LLMResponseMeta(
                sent_at=now,
                input_tokens=5,
                output_tokens=3,
                latency_ms=100,
            ),
        ),
    ]

    # Convert to new format and back
    new_messages = convert_legacy_to_new(original_messages)
    converted_back = convert_new_to_legacy(new_messages)

    assert len(converted_back) == 2
    assert isinstance(converted_back[0], AgentUserMessage)
    assert isinstance(converted_back[1], AgentAssistantMessage)

    assert converted_back[0].content == "Hello"
    assert converted_back[1].content == "Hi there!"
    assert converted_back[1].meta.input_tokens == 5
    assert converted_back[1].meta.output_tokens == 3
