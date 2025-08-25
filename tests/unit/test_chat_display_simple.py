"""Simple tests for chat_display.py to improve coverage."""

from datetime import datetime, timedelta, timezone
from io import StringIO

from rich.console import Console

from lite_agent.chat_display import DisplayConfig, _format_timestamp, _get_local_timezone, build_chat_summary_table, display_chat_summary
from lite_agent.types import AssistantMessageMeta, AssistantTextContent, MessageUsage, NewAssistantMessage, NewUserMessage, UserTextContent


class TestDisplayConfig:
    """Test DisplayConfig class."""

    def test_display_config_default(self):
        """Test DisplayConfig with default values."""
        config = DisplayConfig()

        assert config.console is None
        assert config.show_indices is True
        assert config.show_timestamps is True
        assert config.max_content_length == 1000
        assert config.local_timezone is None

    def test_display_config_custom(self):
        """Test DisplayConfig with custom values."""
        console = Console()
        config = DisplayConfig(
            console=console,
            show_indices=False,
            show_timestamps=False,
            max_content_length=500,
            local_timezone="UTC",
        )

        assert config.console is console
        assert config.show_indices is False
        assert config.show_timestamps is False
        assert config.max_content_length == 500
        assert config.local_timezone == "UTC"


class TestChatDisplayFunctions:
    """Test chat display functions."""

    def test_get_local_timezone(self):
        """Test _get_local_timezone function."""
        tz = _get_local_timezone()
        assert isinstance(tz, timezone)

    def test_format_timestamp_basic(self):
        """Test _format_timestamp function."""
        test_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _format_timestamp(test_time, local_timezone=timezone.utc)

        # Should return a formatted string
        assert isinstance(result, str)
        assert "12:00" in result

    def test_format_timestamp_none_input(self):
        """Test _format_timestamp with None input."""
        result = _format_timestamp(None)

        # Should return a formatted string (uses current time when None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_timestamp_with_timezone_conversion(self):
        """Test _format_timestamp with timezone conversion."""
        utc_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        target_tz = timezone(timedelta(hours=8))  # UTC+8

        result = _format_timestamp(utc_time, local_timezone=target_tz)

        # Should convert timezone
        assert isinstance(result, str)
        assert "20:00" in result  # 12:00 UTC -> 20:00 UTC+8

    def test_build_chat_summary_table_basic(self):
        """Test build_chat_summary_table with basic messages."""
        messages = [
            NewUserMessage(content=[UserTextContent(text="Hello")]),
            NewAssistantMessage(content=[AssistantTextContent(text="Hi there!")]),
        ]

        table = build_chat_summary_table(messages)

        # Should return a table
        assert table is not None
        assert hasattr(table, "title")

    def test_build_chat_summary_table_empty(self):
        """Test build_chat_summary_table with empty messages."""
        messages = []
        table = build_chat_summary_table(messages)

        # Should handle empty messages
        assert table is not None

    def test_build_chat_summary_table_with_metadata(self):
        """Test build_chat_summary_table with message metadata."""
        usage = MessageUsage(input_tokens=10, output_tokens=20)
        messages = [
            NewAssistantMessage(
                content=[AssistantTextContent(text="Response with metadata")],
                meta=AssistantMessageMeta(
                    model="gpt-4",
                    usage=usage,
                    total_time_ms=1500,
                ),
            ),
        ]

        table = build_chat_summary_table(messages)

        # Should handle metadata
        assert table is not None

    def test_display_chat_summary_with_console(self):
        """Test display_chat_summary with custom console."""
        string_io = StringIO()
        console = Console(file=string_io, width=80)

        messages = [
            NewUserMessage(content=[UserTextContent(text="Hello")]),
        ]

        # Should not raise exception
        display_chat_summary(messages, console=console)

        # Should have written something to the console
        output = string_io.getvalue()
        assert len(output) > 0

    def test_display_chat_summary_empty_messages(self):
        """Test display_chat_summary with empty messages."""
        string_io = StringIO()
        console = Console(file=string_io, width=80)

        messages = []

        # Should not raise exception
        display_chat_summary(messages, console=console)

    def test_build_chat_summary_table_dict_messages(self):
        """Test build_chat_summary_table with dict format messages."""
        messages = [
            {"role": "user", "content": "Dict format message"},
            {"role": "assistant", "content": "Dict format response"},
        ]

        table = build_chat_summary_table(messages)  # type: ignore

        # Should handle dict format
        assert table is not None
