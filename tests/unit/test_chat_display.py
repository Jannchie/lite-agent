"""
测试 chat_display 模块的功能
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

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


def test_create_chat_summary_table():
    """测试聊天摘要表格创建。"""
    messages = [
        AgentUserMessage(role="user", content="Hello"),
        AgentAssistantMessage(role="assistant", content="Hi"),
        {"role": "system", "content": "System message"},
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
    ]

    table = build_chat_summary_table(messages)
    assert table.title == "Chat Summary"
    # 表格应该被成功创建，没有异常


def test_create_chat_summary_table_empty():
    """测试空消息列表的摘要表格。"""
    table = build_chat_summary_table([])
    assert table.title == "Chat Summary"
    # 即使是空列表，也应该能创建表格


class TestTimezone:
    """测试时区相关功能"""

    def test_get_local_timezone(self):
        """测试获取本地时区"""
        tz = _get_local_timezone()
        assert isinstance(tz, timezone)

    def test_get_timezone_by_name_local(self):
        """测试根据名称获取本地时区"""
        tz = _get_timezone_by_name("local")
        assert isinstance(tz, timezone)

        tz_upper = _get_timezone_by_name("LOCAL")
        assert isinstance(tz_upper, timezone)

    def test_get_timezone_by_name_utc(self):
        """测试获取UTC时区"""
        tz = _get_timezone_by_name("UTC")
        assert tz == timezone.utc

        tz_lower = _get_timezone_by_name("utc")
        assert tz_lower == timezone.utc

    def test_get_timezone_by_name_offset(self):
        """测试通过偏移量获取时区"""
        tz_plus = _get_timezone_by_name("+8")
        assert tz_plus == timezone(timedelta(hours=8))

        tz_minus = _get_timezone_by_name("-5")
        assert tz_minus == timezone(timedelta(hours=-5))

    def test_get_timezone_by_name_invalid_offset(self):
        """测试无效偏移量返回本地时区"""
        tz = _get_timezone_by_name("+invalid")
        assert isinstance(tz, timezone)

    @patch("lite_agent.chat_display.ZoneInfo", None)
    def test_get_timezone_by_name_no_zoneinfo(self):
        """测试没有zoneinfo时的行为"""
        tz = _get_timezone_by_name("Asia/Shanghai")
        assert isinstance(tz, timezone)

    def test_format_timestamp_current_time(self):
        """测试格式化当前时间"""
        timestamp = _format_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

    def test_format_timestamp_specific_time(self):
        """测试格式化特定时间"""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamp = _format_timestamp(dt)
        assert isinstance(timestamp, str)

    def test_format_timestamp_without_timezone(self):
        """测试格式化没有时区信息的时间"""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=None)  # noqa: DTZ001
        timestamp = _format_timestamp(dt)
        assert isinstance(timestamp, str)

    def test_format_timestamp_custom_format(self):
        """测试自定义时间格式"""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamp = _format_timestamp(dt, format_str="%Y-%m-%d %H:%M")
        assert "2023-01-01" in timestamp


class TestDisplayConfig:
    """测试DisplayConfig类"""

    def test_display_config_default(self):
        """测试默认配置"""
        config = DisplayConfig()
        assert config.console is None
        assert config.show_indices is True
        assert config.show_timestamps is True
        assert config.max_content_length == 1000
        assert config.local_timezone is None

    def test_display_config_custom(self):
        """测试自定义配置"""
        console = Console()
        config = DisplayConfig(
            console=console,
            show_indices=False,
            show_timestamps=False,
            max_content_length=500,
            local_timezone=timezone.utc,
        )
        assert config.console is console
        assert config.show_indices is False
        assert config.show_timestamps is False
        assert config.max_content_length == 500
        assert config.local_timezone == timezone.utc


class TestMessageContext:
    """测试MessageContext类"""

    def test_message_context(self):
        """测试消息上下文"""
        console = Console()
        context = MessageContext(
            console=console,
            index_str="1",
            timestamp_str="12:00:00",
            max_content_length=100,
            truncate_content=lambda x, y: x[:y],
        )
        assert context.console is console
        assert context.index_str == "1"
        assert context.timestamp_str == "12:00:00"
        assert context.max_content_length == 100
        assert callable(context.truncate_content)


class TestDisplayMessages:
    """测试消息显示功能"""

    def test_display_messages_empty(self):
        """测试显示空消息列表"""
        console = Console()
        # 应该不抛出异常
        display_messages([], console=console)

    def test_display_messages_with_different_types(self):
        """测试显示不同类型的消息"""
        console = Console()
        messages = [
            AgentUserMessage(role="user", content="Hello"),
            AgentAssistantMessage(role="assistant", content="Hi there"),
            AgentSystemMessage(role="system", content="System message"),
            NewUserMessage(content=[UserTextContent(text="New user message")]),
            NewAssistantMessage(content=[AssistantTextContent(text="New assistant message")]),
            NewSystemMessage(content="New system message"),
        ]
        # 应该不抛出异常
        display_messages(messages, console=console)

    def test_display_messages_with_config(self):
        """测试使用配置显示消息"""
        console = Console()
        config = DisplayConfig(
            console=console,
            show_indices=False,
            show_timestamps=False,
            max_content_length=50,
        )
        messages = [
            AgentUserMessage(role="user", content="Hello world" * 20),
        ]
        # 应该不抛出异常
        display_messages(messages, config=config)

    def test_display_chat_summary(self):
        """测试显示聊天摘要"""
        console = Console()
        messages = [
            AgentUserMessage(role="user", content="Hello"),
            AgentAssistantMessage(role="assistant", content="Hi there"),
        ]
        # 应该不抛出异常
        display_chat_summary(messages, console=console)

    def test_display_messages_with_function_calls(self):
        """测试显示包含函数调用的消息"""
        console = Console()
        messages = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "Beijing"}',
                "content": "",
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Sunny, 25°C",
            },
        ]
        # 应该不抛出异常
        display_messages(messages, console=console)
