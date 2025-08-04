"""
Additional tests for chat_display.py to improve coverage
"""

from datetime import datetime, timedelta, timezone

import pytest
from rich.console import Console

from lite_agent.chat_display import (
    DisplayConfig,
    MessageContext,
    _format_timestamp,
    _get_local_timezone,
    _get_timezone_by_name,
    build_chat_summary_table,
    display_chat_summary,
    display_messages,
)
from lite_agent.types import (
    AgentAssistantMessage,
    AgentSystemMessage,
    AgentUserMessage,
    AssistantTextContent,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    UserTextContent,
)


class TestChatDisplayAdditional:
    """Additional tests for ChatDisplay functionality"""

    def test_get_timezone_by_name_invalid(self):
        """Test getting timezone by invalid name"""
        tz = _get_timezone_by_name("invalid_timezone")
        # Should fallback to local timezone, not UTC
        local_tz = _get_local_timezone()
        assert tz == local_tz

    def test_format_timestamp_with_different_timezones(self):
        """Test timestamp formatting with different timezones"""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Test with UTC
        result = _format_timestamp(dt, local_timezone=timezone.utc)
        assert "12:00:00" in result

        # Test with different timezone
        offset_tz = timezone(timedelta(hours=5))
        result = _format_timestamp(dt, local_timezone=offset_tz)
        assert "17:00:00" in result  # 12 + 5 hours

    def test_format_timestamp_edge_cases(self):
        """Test timestamp formatting edge cases"""
        # Test with None datetime - returns current time
        result = _format_timestamp(None)
        # Should return a time string in HH:MM:SS format
        assert ":" in result
        assert len(result.split(":")) == 3

        # Test with custom format
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Use UTC timezone to avoid local timezone conversion
        result = _format_timestamp(dt, local_timezone=timezone.utc, format_str="%Y-%m-%d %H:%M")
        assert "2023-01-01 12:00" in result

    def test_display_config_edge_cases(self):
        """Test DisplayConfig with various configurations"""
        # Test with string timezone
        config = DisplayConfig(local_timezone="UTC")
        assert config.local_timezone == "UTC"

        # Test with None values
        config = DisplayConfig(
            console=None,
            local_timezone=None,
        )
        assert config.console is None
        assert config.local_timezone is None

    def test_message_context_creation(self):
        """Test MessageContext creation and properties"""
        console = Console()
        context = MessageContext(
            console=console,
            index_str="[1]",
            timestamp_str="[12:00:00]",
            max_content_length=100,
            truncate_content=lambda x, y: x[:y] + "..." if len(x) > y else x,
        )

        assert context.console is console
        assert context.index_str == "[1]"
        assert context.timestamp_str == "[12:00:00]"
        assert context.max_content_length == 100
        assert callable(context.truncate_content)

        # Test truncate function
        result = context.truncate_content("Hello World", 5)
        assert result == "Hello..."

    def test_build_chat_summary_table_with_complex_messages(self):
        """Test chat summary table with complex message types"""
        messages = [
            AgentUserMessage(role="user", content="Hello"),
            AgentAssistantMessage(role="assistant", content="Hi"),
            AgentSystemMessage(role="system", content="System message"),
            NewUserMessage(content=[UserTextContent(text="New user message")]),
            NewAssistantMessage(content=[AssistantTextContent(text="New assistant message")]),
            NewSystemMessage(content="New system message"),
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "test_func",
                "arguments": "{}",
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result",
            },
            # Test messages with meta data
            {
                "role": "assistant",
                "content": "Assistant with meta",
                "meta": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "latency_ms": 100,
                    "output_time_ms": 50,
                },
            },
        ]

        table = build_chat_summary_table(messages)
        assert table.title == "Chat Summary"

    def test_display_messages_with_various_configs(self):
        """Test display_messages with different configuration options"""
        console = Console()
        messages = [
            AgentUserMessage(role="user", content="Test message"),
            NewAssistantMessage(content=[AssistantTextContent(text="Response")]),
        ]

        # Test with explicit config
        config = DisplayConfig(
            console=console,
            show_indices=False,
            show_timestamps=True,
            max_content_length=50,
            local_timezone=timezone.utc,
        )
        display_messages(messages, config=config)

        # Test with kwargs
        display_messages(
            messages,
            console=console,
            show_indices=True,
            show_timestamps=False,
            max_content_length=200,
            local_timezone="UTC",
        )

    def test_display_messages_empty_list(self):
        """Test display_messages with empty message list"""
        console = Console()
        display_messages([], console=console)

    def test_display_chat_summary_with_empty_messages(self):
        """Test display_chat_summary with empty message list"""
        console = Console()
        display_chat_summary([], console=console)

    def test_get_local_timezone(self):
        """Test local timezone detection"""
        tz = _get_local_timezone()
        assert isinstance(tz, timezone)

    @pytest.mark.parametrize(
        "timezone_name",
        [
            "local",
            "LOCAL",
            "utc",
            "UTC",
            "invalid_name",
        ],
    )
    def test_get_timezone_by_name_cases(self, timezone_name):
        """Test timezone name lookup with various cases"""
        tz = _get_timezone_by_name(timezone_name)
        assert isinstance(tz, timezone)

    def test_display_messages_with_function_calls(self):
        """Test displaying messages with function calls"""
        console = Console()
        messages = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Sunny, 25°C",
            },
        ]

        display_messages(messages, console=console)

    def test_display_messages_with_new_format_complex(self):
        """Test displaying complex new format messages"""
        console = Console()

        # Test NewUserMessage with multiple content types
        user_msg = NewUserMessage(
            content=[
                UserTextContent(text="Hello"),
                UserTextContent(text="World"),
            ],
        )

        # Test NewAssistantMessage with mixed content
        assistant_msg = NewAssistantMessage(
            content=[
                AssistantTextContent(text="Response"),
            ],
        )

        messages = [user_msg, assistant_msg]
        display_messages(messages, console=console)

    def test_format_timestamp_with_none_timezone(self):
        """Test _format_timestamp with None timezone"""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _format_timestamp(dt, local_timezone=None)
        # With None timezone, should convert to local timezone
        # The exact time depends on the system's local timezone
        assert ":" in result
        assert len(result.split(":")) == 3

    def test_display_messages_error_handling(self):
        """Test error handling in display_messages"""
        # Test with invalid message type
        messages = [{"invalid": "message"}]
        console = Console()

        # Should not raise exception
        display_messages(messages, console=console)

    def test_display_config_type_validation(self):
        """Test DisplayConfig type validation in display_messages"""
        console = Console()
        messages = [AgentUserMessage(role="user", content="Test")]

        # Test with invalid types in kwargs (should be filtered out)
        display_messages(
            messages,
            console=console,
            show_indices="invalid",  # Wrong type, should be ignored
            show_timestamps=True,
            max_content_length="invalid",  # Wrong type, should be ignored
        )

    def test_message_with_meta_data_variants(self):
        """Test messages with different meta data structures"""
        from lite_agent.types import AssistantMessageMeta

        messages = [
            # Message with object-style meta
            AgentAssistantMessage(
                content="Test with meta",
                meta=AssistantMessageMeta(
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=100,
                ),
            ),
            # Message with dict-style meta
            {
                "role": "assistant",
                "content": "Test with dict meta",
                "meta": {
                    "input_tokens": 15,
                    "output_tokens": 8,
                    "latency_ms": 150,
                    "output_time_ms": 75,
                },
            },
        ]

        table = build_chat_summary_table(messages)
        assert table.title == "Chat Summary"

        console = Console()
        display_messages(messages, console=console)
