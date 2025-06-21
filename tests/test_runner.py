from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest

from lite_agent.runner import Runner


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
    mock_message = AsyncMock()
    mock_message.model_dump.return_value = {"role": "assistant", "content": "done"}
    mock_agent = Mock()

    async def async_gen(_: object) -> AsyncGenerator[dict, None]:
        yield {"type": "final_message", "message": mock_message, "finish_reason": "stop"}

    mock_agent.stream_async = AsyncMock(side_effect=async_gen)
    runner = Runner(agent=mock_agent)
    gen = runner.run_stream("hello")
    assert hasattr(gen, "__aiter__")
