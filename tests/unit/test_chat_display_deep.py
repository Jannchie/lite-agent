"""Deeper coverage tests for chat_display helpers."""

from datetime import datetime, timedelta, timezone
from io import StringIO

import pytest
from rich.console import Console

from lite_agent import chat_display as cd
from lite_agent.chat_display import (
    DisplayConfig,
    _create_message_context,
    _display_basic_message_stats,
    _display_legacy_message_with_columns,
    _dispatch_message_display,
    display_messages,
    messages_to_string,
)
from lite_agent.types import (
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    AgentAssistantMessage,
    AgentSystemMessage,
    AgentUserMessage,
    LLMResponseMeta,
    MessageUsage,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    UserFileContent,
    UserImageContent,
    UserTextContent,
)


def test_display_messages_rich_content() -> None:
    """display_messages should render images, files, metadata, and tool events."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, no_color=True, width=180)

    user_message = NewUserMessage(
        content=[
            UserImageContent(image_url="https://example.com/img.png"),
            UserFileContent(file_id="file-123", file_name="spec.pdf"),
            UserTextContent(text="Please handle these resources."),
        ],
    )
    assistant_meta = AssistantMessageMeta(
        model="demo-model",
        usage=MessageUsage(input_tokens=4, output_tokens=6),
        latency_ms=21,
        total_time_ms=45,
        output_time_ms=30,
    )
    assistant_message = NewAssistantMessage(
        content=[
            AssistantTextContent(text="Line one\nLine two"),
            AssistantToolCall(call_id="call-1", name="lookup", arguments={"query": "detail"}),
            AssistantToolCall(call_id="call-2", name="fallback", arguments="{not-json"),
            AssistantToolCallResult(call_id="call-2", output="First line\nSecond line", execution_time_ms=12),
        ],
        meta=assistant_meta,
    )
    system_message = NewSystemMessage(content="System update")

    config = DisplayConfig(
        console=console,
        show_indices=True,
        show_timestamps=True,
        show_metadata=True,
        max_content_length=120,
        local_timezone=timezone.utc,
    )

    display_messages([user_message, assistant_message, system_message], config=config)

    output = buffer.getvalue()
    assert "[Image:" in output
    assert "[File:" in output
    assert "Assistant:" in output
    assert "Tokens:" in output
    assert "Call: lookup" in output
    assert "Call: fallback" in output
    assert "Output:" in output
    assert "System:" in output


def test_messages_to_string_empty_list() -> None:
    """messages_to_string on empty conversations should emit the placeholder message."""
    result = messages_to_string([], show_indices=True, show_timestamps=True)
    assert "No messages to display" in result


def test_display_legacy_message_with_columns() -> None:
    """Legacy fallback renderer should print the legacy label."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, no_color=True, width=120)
    _display_legacy_message_with_columns(
        {"legacy": "payload"},
        console=console,
        time_str="12:00:00",
        index_str="#1",
        show_metadata=False,
        max_content_length=40,
        truncate_content=lambda content, limit: content[:limit],
    )
    assert "Legacy" in buffer.getvalue()


def test_get_timezone_by_name_zoneinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure zoneinfo branch returns the expected offset."""

    class FakeZoneInfo:
        def __init__(self, name: str) -> None:
            self.name = name

        def utcoffset(self, dt: datetime) -> timedelta | None:  # noqa: ARG002
            return timedelta(hours=2)

    monkeypatch.setattr(cd, "ZoneInfo", lambda name: FakeZoneInfo(name))
    tz = cd._get_timezone_by_name("Europe/Test")  # noqa: SLF001
    assert tz.utcoffset(datetime.now(timezone.utc)) == timedelta(hours=2)


def _make_context(message: object, *, width: int = 160) -> tuple[cd.MessageContext, StringIO]:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, no_color=True, width=width)
    context = _create_message_context(
        {
            "console": console,
            "index": 1,
            "message": message,
            "max_content_length": 120,
            "truncate_content": lambda text, limit: text[:limit],
            "show_timestamp": True,
            "show_metadata": True,
            "local_timezone": timezone.utc,
        },
    )
    return context, buffer


def test_dispatch_message_display_variants() -> None:
    """_dispatch_message_display should handle new, legacy, dict, and unknown messages."""
    # New user message with multiple content types
    rich_user = NewUserMessage(
        content=[
            UserImageContent(image_url="https://example.com/new.png"),
            UserFileContent(file_id="file-xyz", file_name="guide.pdf"),
            UserTextContent(text="hello"),
        ],
    )
    context, buffer = _make_context(rich_user)
    _dispatch_message_display(rich_user, context)  # noqa: SLF001
    assert "User:" in buffer.getvalue()

    # New assistant message with tool calls and metadata
    assistant_message = NewAssistantMessage(
        content=[
            AssistantTextContent(text="assistant text"),
            AssistantToolCall(call_id="call-json", name="lookup", arguments='{"k": 1}'),
            AssistantToolCall(call_id="call-bad", name="fallback", arguments="{bad-json"),
            AssistantToolCallResult(call_id="call-bad", output="tool output", execution_time_ms=5),
        ],
        meta=AssistantMessageMeta(
            model="demo",
            usage=MessageUsage(input_tokens=1, output_tokens=2),
            latency_ms=10,
            output_time_ms=7,
        ),
    )
    context, buffer = _make_context(assistant_message)
    _dispatch_message_display(assistant_message, context)  # noqa: SLF001
    output = buffer.getvalue()
    assert "Assistant:" in output
    assert "Tokens:" in output
    assert "Call:" in output
    assert "Output:" in output

    # New system message
    system_message = NewSystemMessage(content="system text")
    context, buffer = _make_context(system_message)
    _dispatch_message_display(system_message, context)  # noqa: SLF001
    assert "System:" in buffer.getvalue()

    # Legacy user message
    legacy_user = AgentUserMessage(
        content=[
            UserTextContent(text="legacy user"),
            UserImageContent(image_url="https://example.com/legacy.png"),
        ],
    )
    context, buffer = _make_context(legacy_user)
    _dispatch_message_display(legacy_user, context)  # noqa: SLF001
    assert "legacy user" in buffer.getvalue()

    # Legacy assistant message with full metadata
    legacy_assistant = AgentAssistantMessage(
        content="legacy assistant",
        meta=AssistantMessageMeta(
            model="legacy-model",
            usage=MessageUsage(input_tokens=2, output_tokens=3),
            latency_ms=11,
            output_time_ms=13,
        ),
    )
    context, buffer = _make_context(legacy_assistant)
    _dispatch_message_display(legacy_assistant, context)  # noqa: SLF001
    legacy_output = buffer.getvalue()
    assert "legacy assistant" in legacy_output
    assert "Model:legacy-model" in legacy_output

    # Legacy system message (alias of NewSystemMessage)
    legacy_system = AgentSystemMessage(content="legacy system")
    context, buffer = _make_context(legacy_system)
    _dispatch_message_display(legacy_system, context)  # noqa: SLF001
    assert "legacy system" in buffer.getvalue()

    # Dict-based function call
    dict_call = {"type": "function_call", "name": "lookup", "arguments": '{"q": 1}'}
    context, buffer = _make_context(dict_call)
    _dispatch_message_display(dict_call, context)  # noqa: SLF001
    assert "Call:" in buffer.getvalue()

    # Dict-based function output with execution time
    dict_output = {"type": "function_call_output", "output": "value", "execution_time_ms": 9}
    context, buffer = _make_context(dict_output)
    _dispatch_message_display(dict_output, context)  # noqa: SLF001
    assert "Output:" in buffer.getvalue()

    # Dict-based assistant with meta tokens
    dict_assistant = {
        "role": "assistant",
        "content": "dict assistant",
        "meta": {"model": "dict", "latency_ms": 5, "output_time_ms": 7, "input_tokens": 2, "output_tokens": 3},
    }
    context, buffer = _make_context(dict_assistant)
    _dispatch_message_display(dict_assistant, context)  # noqa: SLF001
    dict_output = buffer.getvalue()
    assert "dict assistant" in dict_output
    assert "Tokens:" in dict_output

    # Dict-based system
    dict_system = {"role": "system", "content": "dict system"}
    context, buffer = _make_context(dict_system)
    _dispatch_message_display(dict_system, context)  # noqa: SLF001
    assert "dict system" in buffer.getvalue()

    # Unknown dict type
    dict_unknown = {"role": "other", "content": "unknown"}
    context, buffer = _make_context(dict_unknown)
    _dispatch_message_display(dict_unknown, context)  # noqa: SLF001
    assert "Unknown" in buffer.getvalue()

    # Unknown object with failing model_dump
    class BrokenMessage:
        def model_dump(self) -> dict:
            raise ValueError("boom")

        def __str__(self) -> str:
            return "broken"

    strange_object = BrokenMessage()
    context, buffer = _make_context(strange_object)
    _dispatch_message_display(strange_object, context)  # noqa: SLF001
    assert "Unknown" in buffer.getvalue()


def test_create_message_context_validation_errors() -> None:
    """_create_message_context should validate inputs."""
    message = NewUserMessage(content=[UserTextContent(text="hello")])
    console = Console(file=StringIO(), force_terminal=False, no_color=True)

    with pytest.raises(TypeError, match="max_content_length must be an integer"):
        _create_message_context(
            {
                "console": console,
                "index": 0,
                "message": message,
                "max_content_length": "not-int",
                "truncate_content": lambda text, limit: text,
                "show_timestamp": False,
                "local_timezone": timezone.utc,
            },
        )

    with pytest.raises(TypeError, match="console must be a Console instance"):
        _create_message_context(
            {
                "console": "not-console",
                "index": 0,
                "message": message,
                "max_content_length": 80,
                "truncate_content": lambda text, limit: text,
            },
        )

    with pytest.raises(TypeError, match="truncate_content must be callable"):
        _create_message_context(
            {
                "console": console,
                "index": 0,
                "message": message,
                "max_content_length": 80,
                "truncate_content": "not-callable",
            },
        )

    with pytest.raises(TypeError, match="local_timezone must be a timezone instance"):
        _create_message_context(
            {
                "console": console,
                "index": 0,
                "message": message,
                "max_content_length": 80,
                "truncate_content": lambda text, limit: text,
                "local_timezone": "UTC",
            },
        )


def test_display_basic_message_stats_with_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the branch that prints the Total row."""
    console_buffer = StringIO()
    console = Console(file=console_buffer, force_terminal=False, no_color=True)

    def fake_analyze_messages(messages: cd.RunnerMessages) -> tuple[dict[str, int], dict[str, int | float]]:
        return (
            {"User": 1, "Assistant": 1, "Total": 2},
            {},
        )

    monkeypatch.setattr(cd, "_analyze_messages", fake_analyze_messages)
    _display_basic_message_stats([], console)  # noqa: SLF001
    assert "Total" in console_buffer.getvalue()


def test_display_messages_creates_default_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """display_messages should instantiate a Console when none is provided."""

    class DummyConsole:
        def __init__(self) -> None:
            self.records: list[str] = []

        def print(self, *args: object, **kwargs: object) -> None:  # noqa: D401
            self.records.append(" ".join(str(arg) for arg in args))

    dummy_console = DummyConsole()
    monkeypatch.setattr(cd, "Console", lambda *args, **kwargs: dummy_console)

    message = NewUserMessage(content=[UserTextContent(text="hello")])
    display_messages([message])
    assert dummy_console.records, "Expect records to be written when using default console."


def test_build_chat_summary_table_with_misc_meta() -> None:
    """Ensure summary statistics handle unknown messages and various meta formats."""
    legacy_meta = AssistantMessageMeta(
        usage=MessageUsage(input_tokens=2, output_tokens=3),
        latency_ms=12,
        output_time_ms=18,
    )
    legacy_message = AgentAssistantMessage(content="legacy response", meta=legacy_meta)

    dict_message = {
        "role": "assistant",
        "content": "dict response",
        "meta": {
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 5,
            "output_time_ms": 6,
        },
    }

    llm_meta = LLMResponseMeta(
        input_tokens=4,
        output_tokens=5,
        latency_ms=20,
        output_time_ms=25,
    )
    llm_message = {
        "role": "assistant",
        "content": "llm meta",
        "meta": llm_meta,
    }

    unknown_message = 42  # type: ignore[arg-type]

    table = cd.build_chat_summary_table([legacy_message, dict_message, llm_message, unknown_message])  # noqa: SLF001
    assert table.title == "Chat Summary"
