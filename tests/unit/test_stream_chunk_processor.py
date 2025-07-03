import pytest

from lite_agent.processors.completion_event_processor import CompletionEventProcessor
from lite_agent.types import AssistantMessage, ToolCall, ToolCallFunction


class DummyDelta:
    def __init__(self, role="assistant"):
        self.role = role


class DummyChoice:
    def __init__(self, delta=None, index=0):
        self.delta = delta or DummyDelta()
        self.index = index


class DummyChunk:
    def __init__(self, id="chunk_id", usage=None):
        self.id = id
        self.usage = usage


class DummyFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class DummyToolCall:
    def __init__(self, id="tool_id", type_="function", function=None, index=0):
        self.id = id
        self.type = type_
        self.function = function or DummyFunction()
        self.index = index


class DummyDeltaToolCall:
    def __init__(self, id=None, type_="function", function=None, index=None):
        self.id = id
        self.type = type_
        self.function = function or DummyFunction()
        self.index = index


@pytest.fixture
def processor():
    return CompletionEventProcessor()


def test_initialize_message_sets_current_message(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    assert processor.current_message is not None
    assert processor.current_message.id == chunk.id
    assert processor.current_message.role == "assistant"


def test_initialize_message_skips_non_assistant(processor):
    chunk = DummyChunk()
    choice = DummyChoice(delta=DummyDelta(role="user"))
    processor.initialize_message(chunk, choice)
    assert processor.is_initialized is False


def test_update_content(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.update_content("hello ")
    processor.update_content("world")
    assert processor.current_message.content == "hello world"


def test_update_content_no_message(processor):
    processor.update_content("should not fail")
    assert processor.is_initialized is False


def test_initialize_tool_calls(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    tool_calls = [DummyToolCall(id="t1"), DummyToolCall(id="t2")]
    processor._initialize_tool_calls(tool_calls)
    assert processor.current_message.tool_calls == []


def test_update_tool_calls(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(function=DummyFunction(arguments="a"))]
    new_calls = [DummyToolCall(function=DummyFunction(arguments="b"))]
    processor._update_tool_calls(new_calls)
    assert processor.current_message.tool_calls[0].function.arguments == "ab"


def test_update_tool_calls_unexpected_type(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(type_="function")]
    new_calls = [DummyToolCall(type_="unexpected")]
    processor._update_tool_calls(new_calls)
    assert processor.current_message.tool_calls[0].type == "function"


def test_update_tool_calls_no_tool_calls(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = None
    processor._update_tool_calls([DummyToolCall()])
    assert processor.current_message.tool_calls is None


def test_update_tool_calls_empty(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = []
    processor._update_tool_calls([])
    assert processor.current_message.tool_calls == []


def test_update_tool_calls_strict_zip(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(function=DummyFunction(arguments="a"))]
    new_calls = [DummyToolCall(function=DummyFunction(arguments="b")), DummyToolCall(function=DummyFunction(arguments="c"))]
    processor._update_tool_calls(new_calls)
    assert processor.current_message.tool_calls[0].function.arguments == "ab"


def test_update_tool_calls_shorter_new_calls(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(function=DummyFunction(arguments="a")), DummyToolCall(function=DummyFunction(arguments="b"))]
    new_calls = [DummyToolCall(function=DummyFunction(arguments="c"))]
    processor._update_tool_calls(new_calls)
    assert processor.current_message.tool_calls[0].function.arguments == "ac"
    assert processor.current_message.tool_calls[1].function.arguments == "b"


def test_update_tool_calls_longer_new_calls(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(function=DummyFunction(arguments="a"))]
    new_calls = [DummyToolCall(function=DummyFunction(arguments="b")), DummyToolCall(function=DummyFunction(arguments="c"))]
    processor._update_tool_calls(new_calls)
    assert processor.current_message.tool_calls[0].function.arguments == "ab"


def test_update_tool_calls_none(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(function=DummyFunction(arguments="a"))]
    processor._update_tool_calls(None)
    assert processor.current_message.tool_calls[0].function.arguments == "a"


def test_update_tool_calls_empty_list(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(function=DummyFunction(arguments="a"))]
    processor._update_tool_calls([])
    assert processor.current_message.tool_calls[0].function.arguments == "a"


def test_update_tool_calls_no_tool_calls_attr(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    del processor.current_message.tool_calls
    processor._update_tool_calls([DummyToolCall()])
    assert hasattr(processor.current_message, "tool_calls")


@pytest.mark.parametrize(
    ("current_type", "new_type", "expected_type"),
    [
        ("function", "function", "function"),
        ("function", "unexpected", "function"),
        ("function", None, "function"),
        ("function", "", "function"),
        (None, None, None),
        (None, "function", "function"),
        (None, "", None),
    ],
)
def test_update_tool_calls_tool_call_type_param(processor, current_type, new_type, expected_type):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [DummyToolCall(type_=current_type)]
    new_calls = [DummyToolCall(type_=new_type)]
    processor._update_tool_calls(new_calls)
    assert processor.current_message.tool_calls[0].type == expected_type


def test_update_tool_calls_no_current_message(processor):
    tool_calls = [DummyDeltaToolCall(id="id1", type_="function", function=DummyFunction(name="f", arguments="a"), index=0)]
    processor.update_tool_calls(tool_calls)


def test_finalize_message(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    msg = processor.current_message
    assert isinstance(msg, AssistantMessage)


def test_update_tool_calls_method(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    tool_calls = [DummyDeltaToolCall(id="id1", type_="function", function=DummyFunction(name="f", arguments="a"), index=0)]
    processor.update_tool_calls(tool_calls)
    assert processor.current_message.tool_calls[0].id == "id1"
    assert processor.current_message.tool_calls[0].function.name == "f"
    assert processor.current_message.tool_calls[0].function.arguments == "a"


def test_update_tool_calls_method_update_existing(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [ToolCall(id="id1", type="function", function=ToolCallFunction(name="f", arguments="a"), index=0)]
    tool_calls = [DummyDeltaToolCall(id=None, type_="function", function=DummyFunction(name="f", arguments="b"), index=0)]
    processor.update_tool_calls(tool_calls)
    assert processor.current_message.tool_calls[0].function.arguments == "ab"


def test_update_tool_calls_method_invalid_index(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = [ToolCall(id="id1", type="function", function=ToolCallFunction(name="f", arguments="a"), index=0)]
    tool_calls = [DummyDeltaToolCall(id=None, type_="function", function=DummyFunction(name="f", arguments="b"), index=1)]
    processor.update_tool_calls(tool_calls)
    assert processor.current_message.tool_calls[0].function.arguments == "a"


def test_update_tool_calls_method_no_tool_calls(processor):
    chunk = DummyChunk()
    choice = DummyChoice()
    processor.initialize_message(chunk, choice)
    processor.current_message.tool_calls = None
    tool_calls = [DummyDeltaToolCall(id=None, type_="function", function=DummyFunction(name="f", arguments="b"), index=0)]
    processor.update_tool_calls(tool_calls)
    assert processor.current_message.tool_calls is None


def test_update_tool_calls_method_no_current_message(processor):
    with pytest.raises(ValueError):  # noqa: PT011
        assert processor.current_message is None
