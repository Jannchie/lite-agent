"""Tests focusing on string conversion helpers in chat_display."""

from datetime import timezone

from lite_agent.chat_display import chat_summary_to_string, messages_to_string
from lite_agent.types import (
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    MessageUsage,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    RunnerMessages,
    UserTextContent,
)


def _sample_messages() -> RunnerMessages:
    """Construct a representative conversation."""
    meta = AssistantMessageMeta(
        model="gpt-test",
        usage=MessageUsage(input_tokens=3, output_tokens=5),
        latency_ms=42,
        total_time_ms=90,
    )
    assistant_message = NewAssistantMessage(
        content=[
            AssistantTextContent(text="Working on your request."),
            AssistantToolCall(call_id="call_1", name="lookup", arguments={"query": "hello"}),
            AssistantToolCallResult(call_id="call_1", output="Result payload", execution_time_ms=15),
        ],
        meta=meta,
    )
    return [
        NewUserMessage(content=[UserTextContent(text="Translate this sentence.")]),
        assistant_message,
        NewSystemMessage(content="Processed successfully."),
    ]


def test_messages_to_string_includes_metadata_and_tools() -> None:
    """messages_to_string should include tool calls, outputs, and metadata when requested."""
    transcript = messages_to_string(
        _sample_messages(),
        show_indices=True,
        show_timestamps=True,
        show_metadata=True,
        local_timezone=timezone.utc,
    )
    assert "Assistant:" in transcript
    assert "lookup" in transcript
    assert "Result payload" in transcript
    assert "Tokens:" in transcript


def test_chat_summary_to_string_variants() -> None:
    """chat_summary_to_string should support both basic and performance-inclusive modes."""
    messages = _sample_messages()

    basic_summary = chat_summary_to_string(messages, include_performance=False)
    assert "Message Summary" in basic_summary

    full_summary = chat_summary_to_string(messages, include_performance=True)
    assert "Chat Summary" in full_summary
    assert "Performance Stats" in full_summary
