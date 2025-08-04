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

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from lite_agent.types import (
    AgentAssistantMessage,
    AgentSystemMessage,
    AgentUserMessage,
    AssistantMessageMeta,
    BasicMessageMeta,
    FlexibleRunnerMessage,
    LLMResponseMeta,
    NewAssistantMessage,
    NewMessage,
    NewSystemMessage,
    NewUserMessage,
    RunnerMessages,
)


@dataclass
class DisplayConfig:
    """消息显示配置。"""

    console: Console | None = None
    show_indices: bool = True
    show_timestamps: bool = True
    max_content_length: int = 1000
    local_timezone: timezone | str | None = None


@dataclass
class MessageContext:
    """消息显示上下文。"""

    console: Console
    index_str: str
    timestamp_str: str
    max_content_length: int
    truncate_content: Callable[[str, int], str]


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
            zone_info = ZoneInfo(timezone_name)
            # 转换为 timezone 对象
            return timezone(zone_info.utcoffset(datetime.now(timezone.utc)) or timedelta(0))
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

    # 统计各种消息类型和 meta 数据
    counts, meta_stats = _analyze_messages(messages)

    # 只显示计数大于0的类型
    for msg_type, count in counts.items():
        if count > 0:
            table.add_row(msg_type, str(count))

    table.add_row("[bold]Total[/bold]", f"[bold]{len(messages)}[/bold]")

    # 添加 meta 数据统计
    _add_meta_stats_to_table(table, meta_stats)

    return table


def _analyze_messages(messages: RunnerMessages) -> tuple[dict[str, int], dict[str, int | float]]:
    """
    分析消息并返回统计信息。

    Args:
        messages: 要分析的消息列表

    Returns:
        消息计数和 meta 数据统计信息的元组
    """
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
        _update_message_counts(message, counts)

        # 收集 meta 数据
        if _is_assistant_message(message):
            meta_data = _extract_meta_data(message, total_input_tokens, total_output_tokens, total_latency_ms, total_output_time_ms)
            if meta_data:
                assistant_with_meta_count += 1
                total_input_tokens, total_output_tokens, total_latency_ms, total_output_time_ms = meta_data

    # 转换为正确的类型
    meta_stats_typed: dict[str, int | float] = {
        "total_input_tokens": float(total_input_tokens),
        "total_output_tokens": float(total_output_tokens),
        "total_latency_ms": float(total_latency_ms),
        "total_output_time_ms": float(total_output_time_ms),
        "assistant_with_meta_count": float(assistant_with_meta_count),
    }
    return counts, meta_stats_typed


def _update_message_counts(message: FlexibleRunnerMessage, counts: dict[str, int]) -> None:
    """更新消息计数。"""
    # Handle new message format first
    if isinstance(message, NewUserMessage):
        counts["User"] += 1
    elif isinstance(message, NewAssistantMessage):
        counts["Assistant"] += 1
        # Count tool calls and outputs within the assistant message
        for content_item in message.content:
            if content_item.type == "tool_call":
                counts["Function Call"] += 1
            elif content_item.type == "tool_call_result":
                counts["Function Output"] += 1
    elif isinstance(message, NewSystemMessage):
        counts["System"] += 1
    # Handle legacy message format
    elif isinstance(message, AgentUserMessage) or (isinstance(message, dict) and message.get("role") == "user"):
        counts["User"] += 1
    elif _is_assistant_message(message):
        counts["Assistant"] += 1
    elif isinstance(message, AgentSystemMessage) or (isinstance(message, dict) and message.get("role") == "system"):
        counts["System"] += 1
    elif isinstance(message, dict) and message.get("type") == "function_call":
        counts["Function Call"] += 1
    elif isinstance(message, dict) and message.get("type") == "function_call_output":
        counts["Function Output"] += 1
    else:
        counts["Unknown"] += 1


def _is_assistant_message(message: FlexibleRunnerMessage) -> bool:
    """判断是否为助手消息。"""
    return isinstance(message, (AgentAssistantMessage, NewAssistantMessage)) or (isinstance(message, dict) and message.get("role") == "assistant")


def _extract_meta_data(message: FlexibleRunnerMessage, total_input: int, total_output: int, total_latency: int, total_output_time: int) -> tuple[int, int, int, int] | None:
    """
    从消息中提取 meta 数据。

    Returns:
        更新后的统计数据元组，如果没有 meta 数据则返回 None
    """
    meta = None
    if isinstance(message, NewAssistantMessage) and message.meta:
        # Handle new message format
        meta = message.meta
        if meta.usage:
            if meta.usage.input_tokens is not None:
                total_input += meta.usage.input_tokens
            if meta.usage.output_tokens is not None:
                total_output += meta.usage.output_tokens
        if meta.latency_ms is not None:
            total_latency += meta.latency_ms
        if meta.total_time_ms is not None:
            total_output_time += meta.total_time_ms
        return total_input, total_output, total_latency, total_output_time
    if isinstance(message, AgentAssistantMessage) and message.meta:
        meta = message.meta
    elif isinstance(message, dict) and message.get("meta"):
        meta = message["meta"]  # type: ignore[typeddict-item]

    if not meta:
        return None

    if hasattr(meta, "input_tokens"):
        return _process_object_meta(meta, total_input, total_output, total_latency, total_output_time)
    if isinstance(meta, dict):
        return _process_dict_meta(meta, total_input, total_output, total_latency, total_output_time)

    return None


def _process_object_meta(meta: BasicMessageMeta | LLMResponseMeta | AssistantMessageMeta, total_input: int, total_output: int, total_latency: int, total_output_time: int) -> tuple[int, int, int, int]:
    """处理对象类型的 meta 数据。"""
    # LLMResponseMeta 和 AssistantMessageMeta 都有这些字段
    if isinstance(meta, (LLMResponseMeta, AssistantMessageMeta)):
        if hasattr(meta, "input_tokens") and meta.input_tokens is not None:
            total_input += int(meta.input_tokens)
        if hasattr(meta, "output_tokens") and meta.output_tokens is not None:
            total_output += int(meta.output_tokens)
        if hasattr(meta, "latency_ms") and meta.latency_ms is not None:
            total_latency += int(meta.latency_ms)
        if hasattr(meta, "output_time_ms") and meta.output_time_ms is not None:
            total_output_time += int(meta.output_time_ms)

    return total_input, total_output, total_latency, total_output_time


def _process_dict_meta(meta: dict[str, str | int | float | None], total_input: int, total_output: int, total_latency: int, total_output_time: int) -> tuple[int, int, int, int]:
    """处理字典类型的 meta 数据。"""
    if meta.get("input_tokens") is not None:
        val = meta["input_tokens"]
        if val is not None:
            total_input += int(val)
    if meta.get("output_tokens") is not None:
        val = meta["output_tokens"]
        if val is not None:
            total_output += int(val)
    if meta.get("latency_ms") is not None:
        val = meta["latency_ms"]
        if val is not None:
            total_latency += int(val)
    if meta.get("output_time_ms") is not None:
        val = meta["output_time_ms"]
        if val is not None:
            total_output_time += int(val)

    return total_input, total_output, total_latency, total_output_time


def _add_meta_stats_to_table(table: Table, meta_stats: dict[str, int | float]) -> None:
    """添加 meta 统计信息到表格。"""
    assistant_with_meta_count = meta_stats["assistant_with_meta_count"]
    if assistant_with_meta_count <= 0:
        return

    table.add_row("", "")  # 空行分隔
    table.add_row("[bold cyan]Performance Stats[/bold cyan]", "")

    total_input_tokens = meta_stats["total_input_tokens"]
    total_output_tokens = meta_stats["total_output_tokens"]
    if total_input_tokens > 0 or total_output_tokens > 0:
        total_tokens = total_input_tokens + total_output_tokens
        table.add_row("Total Tokens", f"↑{total_input_tokens}↓{total_output_tokens}={total_tokens}")

    total_latency_ms = meta_stats["total_latency_ms"]
    if assistant_with_meta_count > 0 and total_latency_ms > 0:
        avg_latency = total_latency_ms / assistant_with_meta_count
        table.add_row("Avg Latency", f"{avg_latency:.1f}ms")

    total_output_time_ms = meta_stats["total_output_time_ms"]
    if assistant_with_meta_count > 0 and total_output_time_ms > 0:
        avg_output_time = total_output_time_ms / assistant_with_meta_count
        table.add_row("Avg Output Time", f"{avg_output_time:.1f}ms")


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
    config: DisplayConfig | None = None,
    **kwargs: object,
) -> None:
    """
    以紧凑的单行格式打印消息列表。

    Args:
        messages: 要打印的消息列表
        config: 显示配置，如果为 None 则使用默认配置
        **kwargs: 额外的配置参数，用于向后兼容

    Example:
        >>> from lite_agent.runner import Runner
        >>> from lite_agent.chat_display import display_messages, DisplayConfig
        >>>
        >>> runner = Runner(agent=my_agent)
        >>> # ... add some messages ...
        >>> display_messages(runner.messages)
        >>> # 或者使用自定义配置
        >>> config = DisplayConfig(show_timestamps=False, max_content_length=100)
        >>> display_messages(runner.messages, config=config)
    """
    if config is None:
        # 过滤掉 None 值的 kwargs 并确保类型正确
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None
            and (
                (k == "console" and isinstance(v, Console))
                or (k == "show_indices" and isinstance(v, bool))
                or (k == "show_timestamps" and isinstance(v, bool))
                or (k == "max_content_length" and isinstance(v, int))
                or (k == "local_timezone" and (isinstance(v, (timezone, str)) or v is None))
            )
        }
        config = DisplayConfig(**filtered_kwargs)  # type: ignore[arg-type]

    console = config.console
    if console is None:
        console = Console()

    if not messages:
        console.print("[dim]No messages to display[/dim]")
        return

    # 处理时区参数
    local_timezone = config.local_timezone
    if local_timezone is None:
        local_timezone = _get_local_timezone()
    elif isinstance(local_timezone, str):
        local_timezone = _get_timezone_by_name(local_timezone)

    for i, message in enumerate(messages):
        _display_single_message_compact(
            message,
            index=i if config.show_indices else None,
            console=console,
            max_content_length=config.max_content_length,
            show_timestamp=config.show_timestamps,
            local_timezone=local_timezone,
        )


def _display_single_message_compact(  # noqa: PLR0913
    message: FlexibleRunnerMessage,
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

    # 创建消息上下文
    context_config = {
        "console": console,
        "index": index,
        "message": message,
        "max_content_length": max_content_length,
        "truncate_content": truncate_content,
        "show_timestamp": show_timestamp,
        "local_timezone": local_timezone,
    }
    context = _create_message_context(context_config)

    # 根据消息类型分发处理
    _dispatch_message_display(message, context)


def _create_message_context(context_config: dict[str, FlexibleRunnerMessage | Console | int | bool | timezone | Callable[[str, int], str] | None]) -> MessageContext:
    """创建消息显示上下文。"""
    console = context_config["console"]
    index = context_config.get("index")
    message = context_config["message"]
    max_content_length_val = context_config["max_content_length"]
    if not isinstance(max_content_length_val, int):
        msg = "max_content_length must be an integer"
        raise TypeError(msg)
    max_content_length = max_content_length_val
    truncate_content = context_config["truncate_content"]
    show_timestamp = context_config.get("show_timestamp", False)
    local_timezone = context_config.get("local_timezone")

    # 类型检查
    console_msg = "console must be a Console instance"
    if not isinstance(console, Console):
        raise TypeError(console_msg)

    truncate_msg = "truncate_content must be callable"
    if not callable(truncate_content):
        raise TypeError(truncate_msg)

    timezone_msg = "local_timezone must be a timezone instance"
    if local_timezone is not None and not isinstance(local_timezone, timezone):
        raise TypeError(timezone_msg)

    # 获取时间戳
    timestamp = None
    if show_timestamp:
        # 确保 message 是正确的类型
        valid_types = (
            AgentUserMessage,
            AgentAssistantMessage,
            AgentSystemMessage,
            NewUserMessage,
            NewAssistantMessage,
            NewSystemMessage,
            dict,
        )
        message_time = _extract_message_time(message) if isinstance(message, valid_types) else None
        timestamp = _format_timestamp(message_time, local_timezone=local_timezone if isinstance(local_timezone, timezone) else None)

    timestamp_str = f"[{timestamp}] " if timestamp else ""
    index_str = f"#{index:2d} " if index is not None else ""

    return MessageContext(
        console=console,
        index_str=index_str,
        timestamp_str=timestamp_str,
        max_content_length=max_content_length,
        truncate_content=truncate_content,  # type: ignore[arg-type]
    )


def _extract_message_time(message: FlexibleRunnerMessage | AgentUserMessage | AgentAssistantMessage | dict) -> datetime | None:
    """从消息中提取时间戳。"""
    # Handle new message format first
    if (isinstance(message, NewMessage) and message.meta and message.meta.sent_at) or (isinstance(message, AgentAssistantMessage) and message.meta and message.meta.sent_at):
        return message.meta.sent_at
    if isinstance(message, dict) and message.get("meta") and isinstance(message["meta"], dict):  # type: ignore[typeddict-item]
        sent_at = message["meta"].get("sent_at")  # type: ignore[typeddict-item]
        if isinstance(sent_at, datetime):
            return sent_at
    return None


def _dispatch_message_display(message: FlexibleRunnerMessage, context: MessageContext) -> None:
    """根据消息类型分发显示处理。"""
    # Handle new message format first
    if isinstance(message, NewUserMessage):
        _display_new_user_message_compact(message, context)
    elif isinstance(message, NewAssistantMessage):
        _display_new_assistant_message_compact(message, context)
    elif isinstance(message, NewSystemMessage):
        _display_new_system_message_compact(message, context)
    # Handle legacy message format
    elif isinstance(message, AgentUserMessage):
        _display_user_message_compact_v2(message, context)
    elif isinstance(message, AgentAssistantMessage):
        _display_assistant_message_compact_v2(message, context)
    elif isinstance(message, AgentSystemMessage):
        _display_system_message_compact_v2(message, context)
    elif isinstance(message, dict):
        _display_dict_message_compact_v2(message, context)  # type: ignore[arg-type]
    else:
        _display_unknown_message_compact_v2(message, context)


def _display_user_message_compact_v2(message: AgentUserMessage, context: MessageContext) -> None:
    """打印用户消息的紧凑格式 (v2)。"""
    content = context.truncate_content(str(message.content), context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[blue]User:[/blue]\n{content}")


def _display_assistant_message_compact_v2(message: AgentAssistantMessage, context: MessageContext) -> None:
    """打印助手消息的紧凑格式 (v2)。"""
    content = context.truncate_content(str(message.content), context.max_content_length)

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

    context.console.print(f"{context.timestamp_str}{context.index_str}[green]Assistant:[/green]{meta_info}\n{content}")


def _display_system_message_compact_v2(message: AgentSystemMessage, context: MessageContext) -> None:
    """打印系统消息的紧凑格式 (v2)。"""
    content = context.truncate_content(str(message.content), context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[yellow]System:[/yellow]\n{content}")


def _display_unknown_message_compact_v2(message: FlexibleRunnerMessage, context: MessageContext) -> None:
    """打印未知类型消息的紧凑格式 (v2)。"""
    try:
        content = str(message.model_dump()) if hasattr(message, "model_dump") else str(message)  # type: ignore[attr-defined]
    except Exception:
        content = str(message)

    content = context.truncate_content(content, context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[red]Unknown:[/red]\n{content}")


def _display_dict_message_compact_v2(message: dict, context: MessageContext) -> None:
    """以紧凑格式打印字典消息 (v2)。"""
    message_type = message.get("type")
    role = message.get("role")

    if message_type == "function_call":
        _display_dict_function_call_compact(message, context)
    elif message_type == "function_call_output":
        _display_dict_function_output_compact(message, context)
    elif role == "user":
        _display_dict_user_compact(message, context)
    elif role == "assistant":
        _display_dict_assistant_compact(message, context)
    elif role == "system":
        _display_dict_system_compact(message, context)
    else:
        # 未知类型的字典消息
        content = context.truncate_content(str(message), context.max_content_length)
        context.console.print(f"{context.timestamp_str}{context.index_str}[red]Unknown:[/red]")
        context.console.print(f"  {content}")


def _display_dict_function_call_compact(message: dict, context: MessageContext) -> None:
    """显示字典类型的函数调用消息。"""
    name = str(message.get("name", "unknown"))
    args = str(message.get("arguments", ""))

    args_str = ""
    if args:
        try:
            parsed_args = json.loads(args)
            args_str = f" {parsed_args}"
        except (json.JSONDecodeError, TypeError):
            args_str = f" {args}"

    args_display = context.truncate_content(args_str, context.max_content_length - len(name) - 10)
    context.console.print(f"{context.timestamp_str}{context.index_str}[magenta]Call:[/magenta] {name}")
    if args_display.strip():  # Only show args if they exist
        context.console.print(f"{args_display.strip()}")


def _display_dict_function_output_compact(message: dict, context: MessageContext) -> None:
    """显示字典类型的函数输出消息。"""
    output = context.truncate_content(str(message.get("output", "")), context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[cyan]Output:[/cyan]")
    context.console.print(f"{output}")


def _display_dict_user_compact(message: dict, context: MessageContext) -> None:
    """显示字典类型的用户消息。"""
    content = context.truncate_content(str(message.get("content", "")), context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[blue]User:[/blue]")
    context.console.print(f"{content}")


def _display_dict_assistant_compact(message: dict, context: MessageContext) -> None:
    """显示字典类型的助手消息。"""
    content = context.truncate_content(str(message.get("content", "")), context.max_content_length)

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

    context.console.print(f"{context.timestamp_str}{context.index_str}[green]Assistant:[/green]{meta_info}")
    context.console.print(f"{content}")


def _display_dict_system_compact(message: dict, context: MessageContext) -> None:
    """显示字典类型的系统消息。"""
    content = context.truncate_content(str(message.get("content", "")), context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[yellow]System:[/yellow]")
    context.console.print(f"{content}")


# New message format display functions
def _display_new_user_message_compact(message: NewUserMessage, context: MessageContext) -> None:
    """显示新格式用户消息的紧凑格式。"""
    # Combine all content into a single string
    content_parts = []
    for item in message.content:
        if item.type == "text":
            content_parts.append(item.text)
        elif item.type == "image":
            if item.image_url:
                content_parts.append(f"[Image: {item.image_url}]")
            elif item.file_id:
                content_parts.append(f"[Image: {item.file_id}]")
        elif item.type == "file":
            file_name = item.file_name or item.file_id
            content_parts.append(f"[File: {file_name}]")

    content = " ".join(content_parts)
    content = context.truncate_content(content, context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[blue]User:[/blue]")
    context.console.print(f"{content}")


def _display_new_system_message_compact(message: NewSystemMessage, context: MessageContext) -> None:
    """显示新格式系统消息的紧凑格式。"""
    content = context.truncate_content(message.content, context.max_content_length)
    context.console.print(f"{context.timestamp_str}{context.index_str}[yellow]System:[/yellow]")
    context.console.print(f"{content}")


def _display_new_assistant_message_compact(message: NewAssistantMessage, context: MessageContext) -> None:
    """显示新格式助手消息的紧凑格式。"""
    # Extract text content and tool information
    text_parts = []
    tool_calls = []
    tool_results = []

    for item in message.content:
        if item.type == "text":
            text_parts.append(item.text)
        elif item.type == "tool_call":
            tool_calls.append(item)
        elif item.type == "tool_call_result":
            tool_results.append(item)

    # Display text content first if available
    if text_parts:
        content = " ".join(text_parts)
        content = context.truncate_content(content, context.max_content_length)

        # Add meta data information (使用英文标签)
        meta_info = ""
        if message.meta:
            meta_parts = []
            if message.meta.latency_ms is not None:
                meta_parts.append(f"Latency:{message.meta.latency_ms}ms")
            if message.meta.total_time_ms is not None:
                meta_parts.append(f"Output:{message.meta.total_time_ms}ms")
            if message.meta.usage and message.meta.usage.input_tokens is not None and message.meta.usage.output_tokens is not None:
                total_tokens = message.meta.usage.input_tokens + message.meta.usage.output_tokens
                meta_parts.append(f"Tokens:↑{message.meta.usage.input_tokens}↓{message.meta.usage.output_tokens}={total_tokens}")

            if meta_parts:
                meta_info = f" [dim]({' | '.join(meta_parts)})[/dim]"

        context.console.print(f"{context.timestamp_str}{context.index_str}[green]Assistant:[/green]{meta_info}")
        context.console.print(f"{content}")

    # Display tool calls
    for tool_call in tool_calls:
        args_str = ""
        if tool_call.arguments:
            try:
                parsed_args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
                args_str = f" {parsed_args}"
            except (json.JSONDecodeError, TypeError):
                args_str = f" {tool_call.arguments}"

        args_display = context.truncate_content(args_str, context.max_content_length - len(tool_call.name) - 10)
        context.console.print(f"{context.timestamp_str}{context.index_str}[magenta]Call:[/magenta]")
        context.console.print(f"{tool_call.name}{args_display}")

    # Display tool results
    for tool_result in tool_results:
        output = context.truncate_content(str(tool_result.output), context.max_content_length)
        context.console.print(f"{context.timestamp_str}{context.index_str}[cyan]Output:[/cyan]")
        context.console.print(f"{output}")
