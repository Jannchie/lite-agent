from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import AgentChunk, Runner
from lite_agent.stream_handlers.litellm import FinalMessageChunk
from lite_agent.types import AssistantMessage


class DummyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(model="dummy-model", name="Dummy Agent", instructions="This is a dummy agent for testing.")

    async def stream_async(self, _message) -> AsyncGenerator[AgentChunk, None]:  # type: ignore
        async def async_gen() -> AsyncGenerator[AgentChunk, None]:
            yield FinalMessageChunk(type="final_message", message=AssistantMessage(role="assistant", content="done", id="123", index=0), finish_reason="stop")

        return async_gen()


@pytest.mark.asyncio
async def test_run_until_complete():
    mock_message = AsyncMock()
    mock_message.model_dump.return_value = {"role": "assistant", "content": "done"}
    mock_agent = Mock()

    async def async_gen(_: object) -> AsyncGenerator[dict, None]:
        yield {"type": "final_message", "message": mock_message, "finish_reason": "stop"}

    mock_agent.stream_async = AsyncMock(side_effect=async_gen)
    runner = Runner(agent=mock_agent)
    result = await runner.run_until_complete("hello")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "final_message"
    mock_agent.stream_async.assert_called_once()


@pytest.mark.asyncio
async def test_run_stream():
    runner = Runner(agent=DummyAgent())
    gen = runner.run_stream("hello")

    # run_stream 返回的是 async generator
    results = []
    async for chunk in gen:
        assert isinstance(chunk, dict)
        assert chunk["type"] == "final_message"
        assert chunk["message"].role == "assistant"
        assert chunk["message"].content == "done"
        results.append(chunk)
    assert hasattr(gen, "__aiter__")
    assert len(results) == 1
