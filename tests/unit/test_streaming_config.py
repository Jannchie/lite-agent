"""Test streaming configuration functionality."""

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self):
        super().__init__(
            model="test-model",
            name="Test Agent",
            instructions="Test instructions",
        )

    async def completion(self, messages, record_to_file=None, reasoning=None, streaming=True):
        """Mock completion method that tracks streaming parameter."""
        self.last_streaming_value = streaming

        # Return empty async generator
        async def empty_gen():
            return
            yield  # This line will never be reached, but makes it a generator

        return empty_gen()

    async def responses(self, messages, record_to_file=None, reasoning=None, streaming=True):
        """Mock responses method that tracks streaming parameter."""
        self.last_streaming_value = streaming

        # Return empty async generator
        async def empty_gen():
            return
            yield  # This line will never be reached, but makes it a generator

        return empty_gen()


@pytest.mark.asyncio
async def test_runner_streaming_default():
    """Test that Runner uses streaming=True by default."""
    agent = MockAgent()
    runner = Runner(agent)

    assert runner.streaming is True


@pytest.mark.asyncio
async def test_runner_streaming_explicit_true():
    """Test that Runner can be configured with streaming=True."""
    agent = MockAgent()
    runner = Runner(agent, streaming=True)

    assert runner.streaming is True


@pytest.mark.asyncio
async def test_runner_streaming_explicit_false():
    """Test that Runner can be configured with streaming=False."""
    agent = MockAgent()
    runner = Runner(agent, streaming=False)

    assert runner.streaming is False


@pytest.mark.asyncio
async def test_runner_passes_streaming_to_completion():
    """Test that Runner passes streaming parameter to agent.completion."""
    agent = MockAgent()

    # Test streaming=True
    runner = Runner(agent, streaming=True)
    async for _ in runner.run("test"):
        pass
    assert agent.last_streaming_value is True

    # Test streaming=False
    runner = Runner(agent, streaming=False)
    async for _ in runner.run("test"):
        pass
    assert agent.last_streaming_value is False


@pytest.mark.asyncio
async def test_runner_passes_streaming_to_responses():
    """Test that Runner passes streaming parameter to agent.responses."""
    agent = MockAgent()

    # Test streaming=True with responses API
    runner = Runner(agent, api="responses", streaming=True)
    async for _ in runner.run("test"):
        pass
    assert agent.last_streaming_value is True

    # Test streaming=False with responses API
    runner = Runner(agent, api="responses", streaming=False)
    async for _ in runner.run("test"):
        pass
    assert agent.last_streaming_value is False
