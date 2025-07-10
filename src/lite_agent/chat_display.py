"""
Chat display utilities for lite-agent.

This module provides utilities to beautifully display chat history using the rich library.
It supports all message types including user messages, assistant messages, function calls,
and function call outputs.
"""

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from lite_agent.types import (
    AgentAssistantMessage,
    AgentFunctionCallOutput,
    AgentFunctionToolCallMessage,
    AgentSystemMessage,
    AgentUserMessage,
    RunnerMessages,
)


def _get_local_timezone() -> timezone:
    """
    检测并返回用户本地时区。

    Returns:
        用户的本地时区对象
    """
    # 获取本地时区偏移（秒）
    offset_seconds = -time.timezone if time.daylight == 0 else -time.altzone
    # 转换为 timezone 对象
    return timezone(timedelta(seconds=offset_seconds))


def _get_timezone_by_name(timezone_name: str) -> timezone:  # noqa: PLR0911
    """
    根据时区名称获取时区对象。

    Args:
        timezone_name: 时区名称，支持：
            - "local": 自动检测本地时区
            - "UTC": UTC 时区
            - "+8", "-5": UTC 偏移量（小时）
            - "Asia/Shanghai", "America/New_York": IANA 时区名称（需要 zoneinfo）

    Returns:
        对应的时区对象
    """
    if timezone_name.lower() == "local":
        return _get_local_timezone()
    if timezone_name.upper() == "UTC":
        return timezone.utc
    if timezone_name.startswith(("+", "-")):
        # 解析 UTC 偏移量，如 "+8", "-5"
        try:
            hours = int(timezone_name)
            return timezone(timedelta(hours=hours))
        except ValueError:
            return _get_local_timezone()
    # 尝试使用 zoneinfo (Python 3.9+)
    elif ZoneInfo is not None:
        try:
            return ZoneInfo(timezone_name)
        except Exception:
            # 如果不支持 zoneinfo，返回本地时区
            return _get_local_timezone()
    else:
        return _get_local_timezone()


def _format_timestamp(
    dt: datetime | None = None,
    *,
    local_timezone: timezone | None = None,
    format_str: str = "%H:%M:%S",
) -> str:
    """
    格式化时间戳，自动转换为本地时区。

    Args:
        dt: 要格式化的 datetime 对象，如果为 None 则使用当前时间
        local_timezone: 本地时区，如果为 None 则自动检测
        format_str: 时间格式字符串

    Returns:
        格式化后的时间字符串
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    if local_timezone is None:
        local_timezone = _get_local_timezone()

    # 如果 datetime 对象没有时区信息，假设为 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # 转换到本地时区
    local_dt = dt.astimezone(local_timezone)
    return local_dt.strftime(format_str)


def display_chat_history(  # noqa: PLR0913
    messages: RunnerMessages,
    *,
    console: Console | None = None,
    show_timestamps: bool = True,
    show_indices: bool = True,
    chat_width: int = 80,
    local_timezone: timezone | str | None = None,
) -> None:
    """
    使用 rich 库美观地显示聊天记录。

    Args:
        messages: 要渲染的消息列表
        console: Rich Console 实例，如果为 None 则创建新的
        show_timestamps: 是否显示时间戳
        show_indices: 是否显示消息索引
        chat_width: 聊天气泡的最大宽度
        local_timezone: 本地时区，支持 timezone 对象或字符串（如 "local", "UTC", "+8", "Asia/Shanghai"），如果为 None 则自动检测

    Example:
        >>> from lite_agent.runner import Runner
        >>> from lite_agent.chat_display import display_chat_history
        >>>
        >>> runner = Runner(agent=my_agent)
        >>> # ... add some messages ...
        >>> display_chat_history(runner.messages)
    """
    if console is None:
        console = Console()

    if not messages:
        console.print("[dim]No messages to display[/dim]")
        return

    # 处理时区参数
    if local_timezone is None:
        local_timezone = _get_local_timezone()
    elif isinstance(local_timezone, str):
        local_timezone = _get_timezone_by_name(local_timezone)

    console.print(f"\n[bold blue]Chat History[/bold blue] ([dim]{len(messages)} messages[/dim])\n")

    for i, message in enumerate(messages):
        _render_single_message(
            message,
            index=i if show_indices else None,
            console=console,
            show_timestamp=show_timestamps,
            chat_width=chat_width,
            local_timezone=local_timezone,
        )


def _render_single_message(  # noqa: PLR0913, C901
    message: object,
    *,
    index: int | None = None,
    console: Console,
    show_timestamp: bool = True,
    chat_width: int = 80,
    local_timezone: timezone | None = None,
) -> None:
    """渲染单个消息。"""
    timestamp = None
    if show_timestamp:
        # 尝试从消息中获取时间戳，否则使用当前时间
        message_time = None
        if isinstance(message, AgentAssistantMessage) and message.meta and message.meta.sent_at:
            message_time = message.meta.sent_at
        elif isinstance(message, dict) and message.get("meta") and isinstance(message["meta"], dict):
            sent_at = message["meta"].get("sent_at")
            if isinstance(sent_at, datetime):
                message_time = sent_at

        timestamp = _format_timestamp(message_time, local_timezone=local_timezone)

    # 处理不同类型的消息
    if isinstance(message, AgentUserMessage):
        _render_user_message(message, index, console, timestamp, chat_width)
    elif isinstance(message, AgentAssistantMessage):
        _render_assistant_message(message, index, console, timestamp, chat_width)
    elif isinstance(message, AgentSystemMessage):
        _render_system_message(message, index, console, timestamp, chat_width)
    elif isinstance(message, AgentFunctionToolCallMessage):
        _render_function_call_message(message, index, console, timestamp, chat_width)
    elif isinstance(message, AgentFunctionCallOutput):
        _render_function_output_message(message, index, console, timestamp, chat_width)
    elif isinstance(message, dict):
        _render_dict_message(message, index, console, timestamp, chat_width)
    else:
        _render_unknown_message(message, index, console, timestamp, chat_width)


def _render_user_message(
    message: AgentUserMessage,
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染用户消息 - 靠右显示的蓝色气泡。"""
    content = str(message.content)  # 显示完整内容，不截断

    title_parts = ["User"]
    if index is not None:
        title_parts.append(f"#{index}")
    if timestamp:
        title_parts.append(f"[dim]{timestamp}[/dim]")

    title = " ".join(title_parts)

    # 计算内容的实际宽度，用于气泡大小
    content_width = min(len(content) + 4, chat_width)  # +4 for padding
    bubble_width = max(content_width, 20)  # 最小宽度

    # 创建用户消息气泡 - 靠右
    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style="blue",
        padding=(0, 1),
        width=bubble_width,
    )

    # 用户消息靠右
    console.print(panel, justify="right")


def _render_assistant_message(
    message: AgentAssistantMessage,
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染助手消息 - 靠左显示的绿色气泡。"""
    content = message.content  # 显示完整内容，不截断

    title_parts = ["Assistant"]
    if index is not None:
        title_parts.append(f"#{index}")
    if timestamp:
        title_parts.append(f"[dim]{timestamp}[/dim]")

    # 添加 meta 数据信息（使用英文标签）
    if message.meta:
        meta_parts = []
        if message.meta.latency_ms is not None:
            meta_parts.append(f"Latency:{message.meta.latency_ms}ms")
        if message.meta.output_time_ms is not None:
            meta_parts.append(f"Output:{message.meta.output_time_ms}ms")
        if message.meta.input_tokens is not None and message.meta.output_tokens is not None:
            total_tokens = message.meta.input_tokens + message.meta.output_tokens
            meta_parts.append(f"Tokens:↑{message.meta.input_tokens}↓{message.meta.output_tokens}={total_tokens}")

        if meta_parts:
            title_parts.append(f"[dim]({' | '.join(meta_parts)})[/dim]")

    title = " ".join(title_parts)

    # 如果有 meta 数据，可能需要稍微增加宽度来容纳更长的标题
    min_width_for_meta = len(title) - 20 if message.meta else 20  # 减去颜色标记的长度
    content_width = min(len(content) + 4, chat_width)  # +4 for padding
    bubble_width = max(content_width, min_width_for_meta, 20)  # 最小宽度

    # 创建助手消息气泡 - 靠左
    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style="green",
        padding=(0, 1),
        width=bubble_width,
    )

    # 助手消息靠左
    console.print(panel)


def _render_system_message(
    message: AgentSystemMessage,
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染系统消息 - 居中显示的黄色气泡。"""
    content = message.content  # 显示完整内容，不截断

    title_parts = ["System"]
    if index is not None:
        title_parts.append(f"#{index}")
    if timestamp:
        title_parts.append(f"[dim]{timestamp}[/dim]")

    title = " ".join(title_parts)

    # 系统消息居中显示，使用较小的宽度
    console.print(
        Panel(
            content,
            title=title,
            title_align="center",
            border_style="yellow",
            padding=(0, 1),
            width=min(len(content) + 10, chat_width),
        ),
        justify="center",
    )


def _render_function_call_message(
    message: AgentFunctionToolCallMessage,
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染函数调用消息 - 靠左显示的紫色气泡。"""
    title_parts = ["Function Call"]
    if index is not None:
        title_parts.append(f"#{index}")
    if timestamp:
        title_parts.append(f"[dim]{timestamp}[/dim]")

    title = " ".join(title_parts)

    # 创建表格显示函数调用详情
    table = Table(show_header=False, box=None, padding=0)
    table.add_column("Field", style="cyan", width=12)
    table.add_column("Value", style="white")

    table.add_row("Name:", f"[bold]{message.name}[/bold]")
    table.add_row("Call ID:", f"[dim]{message.call_id}[/dim]")

    if message.arguments:
        # 尝试格式化 JSON 参数 - 显示完整内容
        try:
            parsed_args = json.loads(message.arguments)
            formatted_args = json.dumps(parsed_args, indent=2, ensure_ascii=False)
            syntax = Syntax(formatted_args, "json", theme="monokai", line_numbers=False)
            table.add_row("Arguments:", syntax)
        except (json.JSONDecodeError, TypeError):
            table.add_row("Arguments:", message.arguments)

    # 函数调用消息靠左
    console.print(
        Panel(
            table,
            title=title,
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
            width=min(chat_width, 100),
        ),
    )


def _render_function_output_message(
    message: AgentFunctionCallOutput,
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染函数输出消息 - 靠左显示的青色气泡。"""
    title_parts = ["Function Output"]
    if index is not None:
        title_parts.append(f"#{index}")
    if timestamp:
        title_parts.append(f"[dim]{timestamp}[/dim]")

    title = " ".join(title_parts)

    output_content = message.output  # 显示完整内容，不截断

    # 创建表格显示函数输出详情
    table = Table(show_header=False, box=None, padding=0)
    table.add_column("Field", style="cyan", width=12)
    table.add_column("Value", style="white")

    table.add_row("Call ID:", f"[dim]{message.call_id}[/dim]")
    table.add_row("Output:", output_content)

    # 函数输出消息靠左
    console.print(
        Panel(
            table,
            title=title,
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            width=min(chat_width, 100),
        ),
    )


def _render_role_based_dict_message(  # noqa: PLR0913
    *,
    message: dict[str, object],
    role: str,
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染基于角色的字典消息。"""
    content = str(message.get("content", ""))  # 显示完整内容，不截断

    title_parts = []
    if role == "user":
        title_parts = ["User"]
        border_style = "blue"
        # 用户消息靠右
        content_width = min(len(content) + 4, chat_width)
        bubble_width = max(content_width, 20)
        if index is not None:
            title_parts.append(f"#{index}")
        if timestamp:
            title_parts.append(f"[dim]{timestamp}[/dim]")

        panel = Panel(
            content,
            title=" ".join(title_parts),
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
            width=bubble_width,
        )
        console.print(panel, justify="right")
    elif role == "assistant":
        title_parts = ["Assistant"]
        border_style = "green"
        if index is not None:
            title_parts.append(f"#{index}")
        if timestamp:
            title_parts.append(f"[dim]{timestamp}[/dim]")

        # 尝试从字典中提取 meta 数据
        meta = message.get("meta")
        if meta and isinstance(meta, dict):
            meta_parts = []
            if meta.get("latency_ms") is not None:
                meta_parts.append(f"Latency:{meta['latency_ms']}ms")
            if meta.get("output_time_ms") is not None:
                meta_parts.append(f"Output:{meta['output_time_ms']}ms")
            if meta.get("input_tokens") is not None and meta.get("output_tokens") is not None:
                total_tokens = meta["input_tokens"] + meta["output_tokens"]
                meta_parts.append(f"Tokens:↑{meta['input_tokens']}↓{meta['output_tokens']}={total_tokens}")

            if meta_parts:
                title_parts.append(f"[dim]({' | '.join(meta_parts)})[/dim]")

        title = " ".join(title_parts)
        min_width_for_meta = len(title) - 20 if meta else 20  # 减去颜色标记的长度
        content_width = min(len(content) + 4, chat_width)
        bubble_width = max(content_width, min_width_for_meta, 20)

        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
            width=bubble_width,
        )
        # 助手消息靠左
        console.print(panel)
    else:  # system
        title_parts = ["System"]
        border_style = "yellow"
        if index is not None:
            title_parts.append(f"#{index}")
        if timestamp:
            title_parts.append(f"[dim]{timestamp}[/dim]")

        # 系统消息居中
        console.print(
            Panel(
                content,
                title=" ".join(title_parts),
                title_align="center",
                border_style=border_style,
                padding=(0, 1),
                width=min(len(content) + 10, chat_width),
            ),
            justify="center",
        )


def _render_dict_message(
    message: dict[str, object],
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染字典格式的消息。"""
    message_type = message.get("type")
    role = message.get("role")

    if message_type == "function_call":
        # 创建临时 AgentFunctionToolCallMessage 对象进行渲染
        temp_message = AgentFunctionToolCallMessage(
            type="function_call",
            call_id=str(message.get("call_id", "")),
            name=str(message.get("name", "unknown")),
            arguments=str(message.get("arguments", "")),
        )
        _render_function_call_message(temp_message, index, console, timestamp, chat_width)
    elif message_type == "function_call_output":
        # 创建临时 AgentFunctionCallOutput 对象进行渲染
        temp_message = AgentFunctionCallOutput(
            type="function_call_output",
            call_id=str(message.get("call_id", "")),
            output=str(message.get("output", "")),
        )
        _render_function_output_message(temp_message, index, console, timestamp, chat_width)
    elif role in ["user", "assistant", "system"]:
        _render_role_based_dict_message(
            message=message,
            role=str(role),
            index=index,
            console=console,
            timestamp=timestamp,
            chat_width=chat_width,
        )
    else:
        _render_unknown_message(message, index, console, timestamp, chat_width)


def _render_unknown_message(
    message: object,
    index: int | None,
    console: Console,
    timestamp: str | None,
    chat_width: int,
) -> None:
    """渲染未知类型的消息 - 居中显示的红色气泡。"""
    title_parts = ["Unknown"]
    if index is not None:
        title_parts.append(f"#{index}")
    if timestamp:
        title_parts.append(f"[dim]{timestamp}[/dim]")

    title = " ".join(title_parts)

    # 尝试将消息转换为可读格式 - 显示完整内容
    try:
        content = str(message.model_dump()) if hasattr(message, "model_dump") else str(message)  # type: ignore[attr-defined]
    except Exception:
        content = str(message)

    console.print(
        Panel(
            content,
            title=title,
            title_align="center",
            border_style="red",
            padding=(0, 1),
            width=min(len(content) + 10, chat_width),
        ),
        justify="center",
    )


def build_chat_summary_table(messages: RunnerMessages) -> Table:
    """
    创建聊天记录摘要表格。

    Args:
        messages: 要汇总的消息列表

    Returns:
        Rich Table 对象，包含消息统计信息
    """
    table = Table(title="Chat Summary")
    table.add_column("Message Type", style="cyan")
    table.add_column("Count", justify="right", style="green")

    # 统计各种消息类型
    counts = {
        "User": 0,
        "Assistant": 0,
        "System": 0,
        "Function Call": 0,
        "Function Output": 0,
        "Unknown": 0,
    }

    # 统计 meta 数据
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0
    total_output_time_ms = 0
    assistant_with_meta_count = 0

    for message in messages:
        if isinstance(message, AgentUserMessage) or (isinstance(message, dict) and message.get("role") == "user"):
            counts["User"] += 1
        elif isinstance(message, AgentAssistantMessage) or (isinstance(message, dict) and message.get("role") == "assistant"):
            counts["Assistant"] += 1

            # 收集 meta 数据
            meta = None
            if isinstance(message, AgentAssistantMessage) and message.meta:
                meta = message.meta
            elif isinstance(message, dict) and message.get("meta"):
                meta = message["meta"]

            if meta:
                assistant_with_meta_count += 1
                if hasattr(meta, "input_tokens"):
                    if meta.input_tokens is not None:
                        total_input_tokens += meta.input_tokens
                    if meta.output_tokens is not None:
                        total_output_tokens += meta.output_tokens
                    if meta.latency_ms is not None:
                        total_latency_ms += meta.latency_ms
                    if meta.output_time_ms is not None:
                        total_output_time_ms += meta.output_time_ms
                elif isinstance(meta, dict):
                    if meta.get("input_tokens") is not None:
                        total_input_tokens += meta["input_tokens"]
                    if meta.get("output_tokens") is not None:
                        total_output_tokens += meta["output_tokens"]
                    if meta.get("latency_ms") is not None:
                        total_latency_ms += meta["latency_ms"]
                    if meta.get("output_time_ms") is not None:
                        total_output_time_ms += meta["output_time_ms"]

        elif isinstance(message, AgentSystemMessage) or (isinstance(message, dict) and message.get("role") == "system"):
            counts["System"] += 1
        elif isinstance(message, AgentFunctionToolCallMessage) or (isinstance(message, dict) and message.get("type") == "function_call"):
            counts["Function Call"] += 1
        elif isinstance(message, AgentFunctionCallOutput) or (isinstance(message, dict) and message.get("type") == "function_call_output"):
            counts["Function Output"] += 1
        else:
            counts["Unknown"] += 1

    # 只显示计数大于0的类型
    for msg_type, count in counts.items():
        if count > 0:
            table.add_row(msg_type, str(count))

    table.add_row("[bold]Total[/bold]", f"[bold]{len(messages)}[/bold]")

    # 添加 meta 数据统计（使用文字而非 emoji）
    if assistant_with_meta_count > 0:
        table.add_row("", "")  # 空行分隔
        table.add_row("[bold cyan]Performance Stats[/bold cyan]", "")

        if total_input_tokens > 0 or total_output_tokens > 0:
            total_tokens = total_input_tokens + total_output_tokens
            table.add_row("Total Tokens", f"↑{total_input_tokens}↓{total_output_tokens}={total_tokens}")

        if assistant_with_meta_count > 0 and total_latency_ms > 0:
            avg_latency = total_latency_ms / assistant_with_meta_count
            table.add_row("Avg Latency", f"{avg_latency:.1f}ms")

        if assistant_with_meta_count > 0 and total_output_time_ms > 0:
            avg_output_time = total_output_time_ms / assistant_with_meta_count
            table.add_row("Avg Output Time", f"{avg_output_time:.1f}ms")

    return table


def display_chat_summary(messages: RunnerMessages, *, console: Console | None = None) -> None:
    """
    打印聊天记录摘要。

    Args:
        messages: 要汇总的消息列表
        console: Rich Console 实例，如果为 None 则创建新的
    """
    if console is None:
        console = Console()

    summary_table = build_chat_summary_table(messages)
    console.print(summary_table)


def display_messages(
    messages: RunnerMessages,
    *,
    console: Console | None = None,
    show_indices: bool = True,
    show_timestamps: bool = True,
    max_content_length: int = 1000,
    local_timezone: timezone | str | None = None,
) -> None:
    """
    以紧凑的单行格式打印消息列表。

    Args:
        messages: 要打印的消息列表
        console: Rich Console 实例，如果为 None 则创建新的
        show_indices: 是否显示消息索引
        show_timestamps: 是否显示时间戳
        max_content_length: 内容的最大显示长度，超过会被截断，默认1000（不截断）
        local_timezone: 本地时区，支持 timezone 对象或字符串（如 "local", "UTC", "+8", "Asia/Shanghai"），如果为 None 则自动检测

    Example:
        >>> from lite_agent.runner import Runner
        >>> from lite_agent.chat_display import display_messages
        >>>
        >>> runner = Runner(agent=my_agent)
        >>> # ... add some messages ...
        >>> display_messages(runner.messages)
    """
    if console is None:
        console = Console()

    if not messages:
        console.print("[dim]No messages to display[/dim]")
        return

    # 处理时区参数
    if local_timezone is None:
        local_timezone = _get_local_timezone()
    elif isinstance(local_timezone, str):
        local_timezone = _get_timezone_by_name(local_timezone)

    for i, message in enumerate(messages):
        _display_single_message_compact(
            message,
            index=i if show_indices else None,
            console=console,
            max_content_length=max_content_length,
            show_timestamp=show_timestamps,
            local_timezone=local_timezone,
        )


def _display_single_message_compact(  # noqa: PLR0913
    message: object,
    *,
    index: int | None = None,
    console: Console,
    max_content_length: int = 100,
    show_timestamp: bool = False,
    local_timezone: timezone | None = None,
) -> None:
    """以紧凑格式打印单个消息。"""

    def truncate_content(content: str, max_length: int) -> str:
        """截断内容并添加省略号。"""
        if len(content) <= max_length:
            return content
        return content[: max_length - 3] + "..."

    # 获取时间戳
    timestamp = None
    if show_timestamp:
        message_time = None
        if isinstance(message, AgentAssistantMessage) and message.meta and message.meta.sent_at:
            message_time = message.meta.sent_at
        elif isinstance(message, dict) and message.get("meta") and isinstance(message["meta"], dict):
            sent_at = message["meta"].get("sent_at")
            if isinstance(sent_at, datetime):
                message_time = sent_at

        timestamp = _format_timestamp(message_time, local_timezone=local_timezone)

    timestamp_str = f"[{timestamp}] " if timestamp else ""
    index_str = f"#{index:2d} " if index is not None else ""

    # 处理不同类型的消息
    if isinstance(message, AgentUserMessage):
        _display_user_message_compact(message, index_str, timestamp_str, console, max_content_length, truncate_content)
    elif isinstance(message, AgentAssistantMessage):
        _display_assistant_message_compact(message, index_str, timestamp_str, console, max_content_length, truncate_content)
    elif isinstance(message, AgentSystemMessage):
        _display_system_message_compact(message, index_str, timestamp_str, console, max_content_length, truncate_content)
    elif isinstance(message, AgentFunctionToolCallMessage):
        _display_function_call_message_compact(message, index_str, timestamp_str, console, max_content_length, truncate_content)
    elif isinstance(message, AgentFunctionCallOutput):
        _display_function_output_message_compact(message, index_str, timestamp_str, console, max_content_length, truncate_content)
    elif isinstance(message, dict):
        _display_dict_message_compact(message, index_str, timestamp_str, console, max_content_length)
    else:
        _display_unknown_message_compact(message, index_str, timestamp_str, console, max_content_length, truncate_content)


def _display_user_message_compact(message: AgentUserMessage, index_str: str, timestamp_str: str, console: Console, max_content_length: int, truncate_content: Callable[[str, int], str]) -> None:
    """打印用户消息的紧凑格式。"""
    content = truncate_content(str(message.content), max_content_length)
    console.print(f"{timestamp_str}{index_str}[blue]User:[/blue]")
    console.print(f"  {content}")


def _display_assistant_message_compact(message: AgentAssistantMessage, index_str: str, timestamp_str: str, console: Console, max_content_length: int, truncate_content: Callable[[str, int], str]) -> None:
    """打印助手消息的紧凑格式。"""
    content = truncate_content(str(message.content), max_content_length)

    # 添加 meta 数据信息（使用英文标签）
    meta_info = ""
    if message.meta:
        meta_parts = []
        if message.meta.latency_ms is not None:
            meta_parts.append(f"Latency:{message.meta.latency_ms}ms")
        if message.meta.output_time_ms is not None:
            meta_parts.append(f"Output:{message.meta.output_time_ms}ms")
        if message.meta.input_tokens is not None and message.meta.output_tokens is not None:
            total_tokens = message.meta.input_tokens + message.meta.output_tokens
            meta_parts.append(f"Tokens:↑{message.meta.input_tokens}↓{message.meta.output_tokens}={total_tokens}")

        if meta_parts:
            meta_info = f" [dim]({' | '.join(meta_parts)})[/dim]"

    console.print(f"{timestamp_str}{index_str}[green]Assistant:[/green]{meta_info}")
    console.print(f"  {content}")


def _display_system_message_compact(message: AgentSystemMessage, index_str: str, timestamp_str: str, console: Console, max_content_length: int, truncate_content: Callable[[str, int], str]) -> None:
    """打印系统消息的紧凑格式。"""
    content = truncate_content(str(message.content), max_content_length)
    console.print(f"{timestamp_str}{index_str}[yellow]System:[/yellow] {content}")


def _display_function_call_message_compact(message: AgentFunctionToolCallMessage, index_str: str, timestamp_str: str, console: Console, max_content_length: int, truncate_content: Callable[[str, int], str]) -> None:
    """打印函数调用消息的紧凑格式。"""
    args_str = ""
    if message.arguments:
        try:
            parsed_args = json.loads(message.arguments)
            args_str = f" {parsed_args}"
        except (json.JSONDecodeError, TypeError):
            args_str = f" {message.arguments}"

    args_display = truncate_content(args_str, max_content_length - len(message.name) - 10)
    console.print(f"{timestamp_str}{index_str}[magenta]Call:[/magenta] {message.name}{args_display}")


def _display_function_output_message_compact(message: AgentFunctionCallOutput, index_str: str, timestamp_str: str, console: Console, max_content_length: int, truncate_content: Callable[[str, int], str]) -> None:
    """打印函数输出消息的紧凑格式。"""
    output = truncate_content(str(message.output), max_content_length)
    console.print(f"{timestamp_str}{index_str}[cyan]Output:[/cyan] {output}")


def _display_unknown_message_compact(message: object, index_str: str, timestamp_str: str, console: Console, max_content_length: int, truncate_content: Callable[[str, int], str]) -> None:
    """打印未知类型消息的紧凑格式。"""
    try:
        content = str(message.model_dump()) if hasattr(message, "model_dump") else str(message)  # type: ignore[attr-defined]
    except Exception:
        content = str(message)

    content = truncate_content(content, max_content_length)
    console.print(f"{timestamp_str}{index_str}[red]Unknown:[/red] {content}")


def _display_dict_message_compact(  # noqa: PLR0913
    message: dict[str, object],
    index_str: str,
    timestamp_str: str,
    console: Console,
    max_content_length: int,
) -> None:
    """以紧凑格式打印字典消息。"""

    def truncate_content(content: str, max_length: int) -> str:
        """截断内容并添加省略号。"""
        if len(content) <= max_length:
            return content
        return content[: max_length - 3] + "..."

    message_type = message.get("type")
    role = message.get("role")

    if message_type == "function_call":
        name = str(message.get("name", "unknown"))
        args = str(message.get("arguments", ""))

        args_str = ""
        if args:
            try:
                parsed_args = json.loads(args)
                args_str = f" {parsed_args}"
            except (json.JSONDecodeError, TypeError):
                args_str = f" {args}"

        args_display = truncate_content(args_str, max_content_length - len(name) - 10)
        console.print(f"{timestamp_str}{index_str}[magenta]Call:[/magenta] {name}")
        if args_display.strip():  # Only show args if they exist
            console.print(f"  {args_display.strip()}")

    elif message_type == "function_call_output":
        output = truncate_content(str(message.get("output", "")), max_content_length)
        console.print(f"{timestamp_str}{index_str}[cyan]Output:[/cyan]")
        console.print(f"  {output}")

    elif role == "user":
        content = truncate_content(str(message.get("content", "")), max_content_length)
        console.print(f"{timestamp_str}{index_str}[blue]User:[/blue]")
        console.print(f"  {content}")

    elif role == "assistant":
        content = truncate_content(str(message.get("content", "")), max_content_length)

        # 添加 meta 数据信息（使用英文标签）
        meta_info = ""
        meta = message.get("meta")
        if meta and isinstance(meta, dict):
            meta_parts = []
            if meta.get("latency_ms") is not None:
                meta_parts.append(f"Latency:{meta['latency_ms']}ms")
            if meta.get("output_time_ms") is not None:
                meta_parts.append(f"Output:{meta['output_time_ms']}ms")
            if meta.get("input_tokens") is not None and meta.get("output_tokens") is not None:
                total_tokens = meta["input_tokens"] + meta["output_tokens"]
                meta_parts.append(f"Tokens:↑{meta['input_tokens']}↓{meta['output_tokens']}={total_tokens}")

            if meta_parts:
                meta_info = f" [dim]({' | '.join(meta_parts)})[/dim]"

        console.print(f"{timestamp_str}{index_str}[green]Assistant:[/green]{meta_info}")
        console.print(f"  {content}")

    elif role == "system":
        content = truncate_content(str(message.get("content", "")), max_content_length)
        console.print(f"{timestamp_str}{index_str}[yellow]System:[/yellow]")
        console.print(f"  {content}")

    else:
        # 未知类型的字典消息
        content = truncate_content(str(message), max_content_length)
        console.print(f"{timestamp_str}{index_str}[red]Unknown:[/red]")
        console.print(f"  {content}")
