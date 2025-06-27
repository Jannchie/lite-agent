from unittest.mock import AsyncMock, MagicMock, Mock

import litellm
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import lite_agent.stream_handlers.litellm as handler_mod
from lite_agent.stream_handlers.litellm import (
    handle_content_and_tool_calls,
    handle_usage_chunk,
    litellm_stream_handler,
)
from lite_agent.types import ToolCall, ToolCallFunction, Usage, UsageChunk


class DummyDelta(Delta):
    def __init__(self, content: str | None = None, tool_calls: list[ToolCall] | None = None):
        super().__init__()
        self.content = content
        self.tool_calls = tool_calls


class DummyChoice(StreamingChoices):
    def __init__(self, delta: DummyDelta | None = None, finish_reason: str | None = None, index: int = 0):
        super().__init__()
        self.delta = delta or DummyDelta()
        self.finish_reason = finish_reason
        self.index = index


class DummyChunk(ModelResponseStream):
    def __init__(self, cid: str = "cid", usage: dict | None = None, choices: list[StreamingChoices] | None = None):
        super().__init__()
        self.id = cid
        self.usage = usage
        self.choices = choices or []


class DummyToolCall:
    def __init__(self, tid: str = "tid", ttype: str = "function", function: ToolCallFunction | None = None, index: int = 0):
        self.id = tid
        self.type = ttype
        self.function = function or DummyFunction()
        self.index = index


class DummyFunction:
    def __init__(self, name="func", arguments="args"):
        self.name = name
        self.arguments = arguments


@pytest.mark.asyncio
async def test_handle_usage_chunk_with_usage():
    processor = MagicMock()
    chunk = DummyChunk(usage={"prompt_tokens": 10})
    processor.handle_usage_info.return_value = Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
    result = await handle_usage_chunk(processor, chunk)
    assert result is not None
    assert result.usage == Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)


@pytest.mark.asyncio
async def test_handle_usage_chunk_without_usage():
    processor = MagicMock()
    chunk = DummyChunk()
    processor.handle_usage_info.return_value = None
    result = await handle_usage_chunk(processor, chunk)
    assert result is None


@pytest.mark.asyncio
async def test_handle_content_and_tool_calls_content_and_tool_calls():
    processor = MagicMock()
    processor.current_message = None
    chunk = DummyChunk()
    choice = DummyChoice()
    delta = DummyDelta(content="hello", tool_calls=[DummyToolCall(function=DummyFunction(arguments="a"))])  # type: ignore
    processor.initialize_message = Mock()
    processor.update_content = Mock()
    processor.update_tool_calls = Mock()
    processor.current_message = MagicMock()
    processor.current_message.tool_calls = [DummyToolCall(tid="tid", function=DummyFunction(name="f", arguments="a"))]  # type: ignore
    results = await handle_content_and_tool_calls(processor, chunk, choice, delta)
    assert any(r.type == "content_delta" for r in results)
    assert any(r.type == "tool_call_delta" for r in results)


@pytest.mark.asyncio
async def test_handle_content_and_tool_calls_no_content_no_tool_calls():
    processor = MagicMock()
    processor.current_message = None
    chunk = DummyChunk()
    choice = DummyChoice()
    delta = DummyDelta(content=None, tool_calls=None)
    processor.initialize_message = Mock()
    results = await handle_content_and_tool_calls(processor, chunk, choice, delta)
    assert results == []


@pytest.mark.asyncio
async def test_chunk_handler_yields_usage(monkeypatch):
    import lite_agent.stream_handlers.litellm as litellm_stream_handler

    chunk = MagicMock(spec=ModelResponseStream)
    chunk.usage = {"prompt_tokens": 10}
    choice = MagicMock(spec=StreamingChoices)
    chunk.choices = [choice]
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([chunk])
    monkeypatch.setattr(litellm_stream_handler, "handle_usage_chunk", AsyncMock(return_value=UsageChunk(type="usage", usage=Usage(prompt_tokens=10))))
    results = []
    async for c in litellm_stream_handler.litellm_stream_handler(resp):
        results.append(c)
        print(c)
    assert any(r.type == "usage" for r in results)


@pytest.mark.asyncio
async def test_chunk_handler_yields_completion_raw(monkeypatch):
    chunk = MagicMock(spec=ModelResponseStream)
    chunk.usage = None
    chunk.choices = []
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([chunk])
    monkeypatch.setattr(handler_mod, "handle_usage_chunk", AsyncMock(return_value=None))
    results = []
    async for c in handler_mod.litellm_stream_handler(resp):
        results.append(c)
    assert any(r.type == "completion_raw" for r in results)


@pytest.mark.asyncio
async def test_handle_content_and_tool_calls_tool_calls_empty():
    processor = MagicMock()
    processor.current_message = MagicMock()
    chunk = DummyChunk()
    choice = DummyChoice()
    delta = DummyDelta(content=None, tool_calls=[])
    processor.initialize_message = Mock()
    processor.update_content = Mock()
    processor.update_tool_calls = Mock()
    processor.current_message.tool_calls = []
    results = await handle_content_and_tool_calls(processor, chunk, choice, delta)
    assert results == []


@pytest.mark.asyncio
async def test_handle_usage_chunk_exception():
    processor = MagicMock()
    chunk = DummyChunk()
    processor.handle_usage_info.side_effect = Exception("fail")
    # 应该抛出异常
    with pytest.raises(Exception, match="fail"):
        await handle_usage_chunk(processor, chunk)


@pytest.mark.asyncio
async def test_litellm_stream_handler_yields_require_confirm():
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    litellm_stream_handler(resp)
