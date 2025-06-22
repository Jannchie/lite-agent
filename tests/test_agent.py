import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from lite_agent.agent import Agent

@pytest.mark.asyncio
async def test_prepare_messages():
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)
    messages = [{"role": "user", "content": "hi"}]
    result = agent.prepare_messages(messages)
    assert result[0]["role"] == "system"
    assert "TestBot" in result[0]["content"]
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "hi"

@pytest.mark.asyncio
async def test_stream_async_success():
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)
    agent.fc.get_tools = MagicMock(return_value=[{"name": "tool1"}])
    fake_resp = MagicMock()
    with patch("lite_agent.agent.litellm.acompletion", new=AsyncMock(return_value=fake_resp)):
        with patch("lite_agent.agent.CustomStreamWrapper", new=lambda x: True):
            # Patch Agent模块作用域下的litellm_stream_handler
            async def fake_async_gen(*args, **kwargs):
                yield "GENERATOR"
            with patch("lite_agent.agent.litellm_stream_handler", new=fake_async_gen):
                with patch("lite_agent.agent.isinstance", new=lambda obj, typ: True):
                    result = await agent.stream_async([{"role": "user", "content": "hi"}])
                    assert hasattr(result, "__aiter__")
                    items = []
                    async for item in result:
                        items.append(item)
                    assert items == ["GENERATOR"]

@pytest.mark.asyncio
async def test_stream_async_typeerror():
    agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.", tools=None)
    agent.fc.get_tools = MagicMock(return_value=[{"name": "tool1"}])
    not_a_stream = object()
    class DummyWrapper: pass
    with patch("lite_agent.agent.litellm.acompletion", new=AsyncMock(return_value=not_a_stream)):
        with patch("lite_agent.agent.CustomStreamWrapper", DummyWrapper):
            with pytest.raises(TypeError, match="Response is not a CustomStreamWrapper"):
                await agent.stream_async([{"role": "user", "content": "hi"}])
