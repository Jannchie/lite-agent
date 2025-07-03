"""
Rich chat history renderer for lite-agent.

This module provides utilities to beautifully render chat history using the rich library.
It supports all message types including user messages, assistant messages, function calls,
and function call outputs.
"""

import json
from datetime import datetime, timezone

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


def print_chat_history(
    messages: RunnerMessages,
    *,
    console: Console | None = None,
    show_timestamps: bool = True,
    show_indices: bool = True,
    chat_width: int = 80,
) -> None:
    """
    使用 rich 库美观地渲染聊天记录。

    Args:
        messages: 要渲染的消息列表
        console: Rich Console 实例，如果为 None 则创建新的
        show_timestamps: 是否显示时间戳
        show_indices: 是否显示消息索引
        chat_width: 聊天气泡的最大宽度

    Example:
        >>> from lite_agent.runner import Runner
        >>> from lite_agent.rich_helpers import render_chat_history
        >>>
        >>> runner = Runner(agent=my_agent)
        >>> # ... add some messages ...
        >>> render_chat_history(runner.messages)
    """
    if console is None:
        console = Console()

    if not messages:
        console.print("[dim]No messages to display[/dim]")
        return

    console.print(f"\n[bold blue]Chat History[/bold blue] ([dim]{len(messages)} messages[/dim])\n")

    for i, message in enumerate(messages):
        _render_single_message(
            message,
            index=i if show_indices else None,
            console=console,
            show_timestamp=show_timestamps,
            chat_width=chat_width,
        )


def _render_single_message(
    message: object,
    *,
    index: int | None = None,
    console: Console,
    show_timestamp: bool = True,
    chat_width: int = 80,
) -> None:
    """渲染单个消息。"""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S") if show_timestamp else None

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

    title_parts = ["👤 User"]
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

    title_parts = ["🤖 Assistant"]
    if index is not None:
        title_parts.append(f"#{index}")
    if timestamp:
        title_parts.append(f"[dim]{timestamp}[/dim]")

    title = " ".join(title_parts)

    # 计算内容的实际宽度，用于气泡大小
    content_width = min(len(content) + 4, chat_width)  # +4 for padding
    bubble_width = max(content_width, 20)  # 最小宽度

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

    title_parts = ["⚙️ System"]
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
    title_parts = ["🛠️ Function Call"]
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
    table.add_row("Call ID:", f"[dim]{message.function_call_id}[/dim]")

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
    title_parts = ["📤 Function Output"]
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
        title_parts = ["👤 User"]
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
        title_parts = ["🤖 Assistant"]
        border_style = "green"
        # 助手消息靠左
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
        # 助手消息靠左
        console.print(panel)
    else:  # system
        title_parts = ["⚙️ System"]
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
            function_call_id=str(message.get("function_call_id", "")),
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
    title_parts = ["❓ Unknown"]
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


def create_chat_summary_table(messages: RunnerMessages) -> Table:
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

    for message in messages:
        if isinstance(message, AgentUserMessage) or (isinstance(message, dict) and message.get("role") == "user"):
            counts["User"] += 1
        elif isinstance(message, AgentAssistantMessage) or (isinstance(message, dict) and message.get("role") == "assistant"):
            counts["Assistant"] += 1
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

    return table


def print_chat_summary(messages: RunnerMessages, *, console: Console | None = None) -> None:
    """
    打印聊天记录摘要。

    Args:
        messages: 要汇总的消息列表
        console: Rich Console 实例，如果为 None 则创建新的
    """
    if console is None:
        console = Console()

    summary_table = create_chat_summary_table(messages)
    console.print(summary_table)
