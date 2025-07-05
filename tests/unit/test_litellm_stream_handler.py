from unittest.mock import MagicMock

import litellm
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import lite_agent.stream_handlers.litellm as handler_mod
from lite_agent.stream_handlers.litellm import (
    litellm_completion_stream_handler,
)
from lite_agent.types import ToolCall, ToolCallFunction


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
async def test_chunk_handler_yields_usage():
    import lite_agent.stream_handlers.litellm as litellm_completion_stream_handler

    chunk = MagicMock(spec=ModelResponseStream)
    chunk.usage = {"prompt_tokens": 10, "completion_tokens": 10}
    choice = MagicMock(spec=StreamingChoices)
    chunk.choices = [choice]
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([chunk])
    results = []
    async for c in litellm_completion_stream_handler.litellm_completion_stream_handler(resp):
        results.append(c)


@pytest.mark.asyncio
async def test_chunk_handler_yields_completion_raw():
    chunk = MagicMock(spec=ModelResponseStream)
    chunk.usage = None
    chunk.choices = []
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    resp.__aiter__.return_value = iter([chunk])
    results = []
    async for c in handler_mod.litellm_completion_stream_handler(resp):
        results.append(c)
    assert any(r.type == "completion_raw" for r in results)


@pytest.mark.asyncio
async def test_litellm_completion_stream_handler_yields_require_confirm():
    resp = MagicMock(spec=litellm.CustomStreamWrapper)
    litellm_completion_stream_handler(resp)
