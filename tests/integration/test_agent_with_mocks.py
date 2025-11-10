"""
Test lite_agent using real mock data generated from basic.py
"""

import asyncio
from unittest.mock import Mock, patch

import pytest
from funcall.decorators import tool

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from tests.utils.mock_openai import create_chat_completion_stream_mock


@tool(require_confirmation=True)
async def get_whether(city: str) -> str:
    """Get the weather for a city."""
    await asyncio.sleep(0.01)  # Reduce sleep time for tests
    return f"The weather in {city} is sunny with a few clouds."


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(0.01)  # Reduce sleep time for tests
    return f"The temperature in {city} is 25°C."


@pytest.mark.asyncio
async def test_agent_with_mock_data():
    """Test the agent using real mock data generated from basic.py."""
    mock_stream = create_chat_completion_stream_mock("tests/mocks/basic/1.jsonl")
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = mock_stream
    mock_client.responses = Mock()

    with patch("lite_agent.client.AsyncOpenAI", return_value=mock_client):
        agent = Agent(
            model="gpt-4.1-nano",
            name="Weather Assistant",
            instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
            tools=[get_whether, get_temperature],
        )
        runner = Runner(agent, api="completion")

        await runner.run_until_complete(
            "What is the weather in New York? And what is the temperature there?",
            includes=["assistant_message", "usage", "function_call", "function_call_output"],
        )

        await runner.run_until_complete(
            None,
            includes=["assistant_message", "usage", "function_call", "function_call_output"],
        )


@pytest.mark.asyncio
async def test_agent_without_mock_data_fails():
    """Test that agent fails gracefully when mock data is missing."""
    # Use a non-existent directory
    mock = create_chat_completion_stream_mock("tests/mocks/nonexistent/file.jsonl")

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = mock
    mock_client.responses = Mock()

    with patch("lite_agent.client.AsyncOpenAI", return_value=mock_client):
        agent = Agent(
            model="gpt-4.1-nano",
            name="Weather Assistant",
            instructions="Test instructions.",
            tools=[get_whether, get_temperature],
        )

        runner = Runner(agent, api="completion")

        # This should raise FileNotFoundError since no mock data exists
        error_raised = False
        try:
            resp = runner.run("What is the weather?")
            async for _ in resp:
                pass
        except FileNotFoundError:
            error_raised = True

        assert error_raised, "Expected FileNotFoundError was not raised"
