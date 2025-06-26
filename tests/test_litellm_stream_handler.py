from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import litellm
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import lite_agent.stream_handlers.litellm as handler_mod
from lite_agent.stream_handlers.litellm import (
    handle_content_and_tool_calls,
    handle_final_message_and_tool_calls,
    handle_usage_chunk,
    litellm_stream_handler,
)
from lite_agent.types import AssistantMessage, ToolCall, ToolCallFunction


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
    processor.handle_usage_info.return_value = {"prompt_tokens": 10}
    result = await handle_usage_chunk(processor, chunk)
    assert result is not None
    assert result["usage"] == {"prompt_tokens": 10}


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
    assert any(r.get("type") == "content_delta" for r in results)
    assert any(r.get("type") == "tool_call_delta" for r in results)


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
async def test_handle_final_message_and_tool_calls_success():
    processor = MagicMock()
    fc = AsyncMock()
    choice = DummyChoice(finish_reason="stop")
    msg = AssistantMessage(id="mid", index=0, role="assistant", content="hi", tool_calls=[ToolCall(id="tid", type="function", function=ToolCallFunction(name="f", arguments="a"), index=0)])
    processor.finalize_message.return_value = msg
    fc.call_function_async.return_value = "result"
    # Patch function_registry 和 get_tool_meta
    fc.function_registry = {"f": lambda: None}
    fc.get_tool_meta = lambda _name: {"require_confirm": False}
    results = await handle_final_message_and_tool_calls(processor, choice, fc)
    assert any(r.get("type") == "final_message" for r in results)
    assert any(r.get("type") == "tool_call_result" for r in results)


@pytest.mark.asyncio
async def test_handle_final_message_and_tool_calls_exception():
    processor = MagicMock()
    fc = AsyncMock()
    choice = DummyChoice(finish_reason="stop")
    msg = AssistantMessage(id="mid", index=0, role="assistant", content="hi", tool_calls=[ToolCall(id="tid", type="function", function=ToolCallFunction(name="f", arguments="a"), index=0)])
    processor.finalize_message.return_value = msg
    fc.call_function_async.side_effect = Exception("fail")
    fc.function_registry = {"f": lambda: None}
    fc.get_tool_meta = lambda _name: {"require_confirm": False}
    results = await handle_final_message_and_tool_calls(processor, choice, fc)
    # 只在 tool_call_result 类型的结果中断言 content
    assert any(r.get("type") == "final_message" for r in results)
    assert any(r.get("type") == "tool_call_result" for r in results)
    assert any("fail" in cast("dict", r)["content"] for r in results if r.get("type") == "tool_call_result")


@pytest.mark.asyncio
async def test_handle_final_message_and_tool_calls_require_confirm():
    processor = MagicMock()
    fc = AsyncMock()
    choice = DummyChoice(finish_reason="stop")
    msg = AssistantMessage(
        id="mid",
        index=0,
        role="assistant",
        content="hi",
        tool_calls=[ToolCall(id="tid", type="function", function=ToolCallFunction(name="f", arguments="a"), index=0)],
    )
    processor.finalize_message.return_value = msg
    fc.function_registry = {"f": lambda: None}
    fc.get_tool_meta = lambda _name: {"require_confirm": True}
    results = await handle_final_message_and_tool_calls(processor, choice, fc)
    assert any(r.get("type") == "require_confirm" for r in results)


# 替换 DummyResp 为 MagicMock(spec=litellm.CustomStreamWrapper) 用于 litellm_stream_handler 测试
@pytest.mark.asyncio
async def test_chunk_handler_yields_usage(monkeypatch):
    import lite_agent.stream_handlers.litellm as litellm_stream_handler

    chunk = MagicMock(spec=ModelResponseStream)
    chunk.usage = {"prompt_tokens": 10}
    choice = MagicMock(spec=StreamingChoices)
    chunk.choices = [choice]
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([chunk])
    monkeypatch.setattr(litellm_stream_handler, "handle_usage_chunk", AsyncMock(return_value={"type": "usage", "usage": {"prompt_tokens": 10}}))
    results = []
    async for c in litellm_stream_handler.litellm_stream_handler(resp):
        results.append(c)
    assert any(r.get("type") == "usage" for r in results)


@pytest.mark.asyncio
async def test_chunk_handler_yields_litellm_raw(monkeypatch):
    chunk = MagicMock(spec=ModelResponseStream)
    chunk.usage = None
    chunk.choices = []
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([chunk])
    monkeypatch.setattr(handler_mod, "handle_usage_chunk", AsyncMock(return_value=None))
    results = []
    async for c in handler_mod.litellm_stream_handler(resp):
        results.append(c)
    assert any(r.get("type") == "litellm_raw" for r in results)


@pytest.mark.asyncio
async def test_chunk_handler_non_model_response_stream(monkeypatch):
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([MagicMock()])
    orig_isinstance = isinstance

    def fake_isinstance(obj, typ):  # noqa: ANN202
        if typ is object:
            return False
        return orig_isinstance(obj, typ)

    monkeypatch.setattr("litellm.types.utils.ModelResponseStream", object)
    monkeypatch.setattr(handler_mod, "handle_usage_chunk", AsyncMock(return_value=None))
    with patch("builtins.isinstance", new=fake_isinstance):
        results = []
        async for chunk in handler_mod.litellm_stream_handler(resp):
            results.append(chunk)
    assert results == []


@pytest.mark.asyncio
async def test_chunk_handler_finish_reason_but_no_current_message(monkeypatch):
    chunk = MagicMock(spec=ModelResponseStream)
    choice = MagicMock(spec=StreamingChoices)
    delta = MagicMock(spec=Delta)
    choice.delta = delta
    choice.finish_reason = "stop"
    chunk.usage = None
    chunk.choices = [choice]
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([chunk])
    monkeypatch.setattr(handler_mod, "handle_usage_chunk", AsyncMock(return_value=None))
    monkeypatch.setattr(handler_mod, "handle_content_and_tool_calls", AsyncMock(return_value=[]))
    monkeypatch.setattr(handler_mod, "handle_final_message_and_tool_calls", AsyncMock(return_value=[{"type": "final_message"}]))
    with patch("lite_agent.processors.StreamChunkProcessor", autospec=True) as MockProc:  # noqa: N806
        proc = MockProc.return_value
        proc.current_message = None
        with patch("lite_agent.processors.StreamChunkProcessor", return_value=proc):
            results = []
            async for c in handler_mod.litellm_stream_handler(resp):
                results.append(c)
    assert any(r.get("type") == "litellm_raw" for r in results)


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
async def test_handle_final_message_and_tool_calls_no_tool_calls():
    processor = MagicMock()
    fc = AsyncMock()
    choice = DummyChoice(finish_reason="stop")
    msg = AssistantMessage(id="mid", index=0, role="assistant", content="hi", tool_calls=None)
    processor.finalize_message.return_value = msg
    results = await handle_final_message_and_tool_calls(processor, choice, fc)
    assert any(r.get("type") == "final_message" for r in results)
    assert not any(r.get("type") == "tool_call" for r in results)


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
