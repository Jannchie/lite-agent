"""
Test lite_agent using real mock data generated from basic.py
"""

import asyncio
from unittest.mock import patch

import pytest
from funcall.decorators import tool

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from tests.utils.mock_litellm import create_litellm_mock


@tool(require_confirmation=True)
async def get_whether(city: str) -> str:
    """Get the weather for a city."""
    await asyncio.sleep(0.01)  # Reduce sleep time for tests
    return f"The weather in {city} is sunny with a few clouds."


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(0.01)  # Reduce sleep time for tests
    return f"The temperature in {city} is 25°C."


agent = Agent(
    model="gpt-4.1-nano",
    name="Weather Assistant",
    instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
    tools=[get_whether, get_temperature],
)

runner = Runner(agent, api="completion")


@pytest.mark.asyncio
async def test_agent_with_mock_data():
    """Test the agent using real mock data generated from basic.py."""
    mock1 = create_litellm_mock("tests/mocks/basic/1.jsonl")
    mock2 = create_litellm_mock("tests/mocks/basic/1.jsonl")  # 使用同一个文件作为示例
    with patch("lite_agent.client.litellm.acompletion", mock1):
        await runner.run_until_complete(
            "What is the weather in New York? And what is the temperature there?",
            includes=["assistant_message", "usage", "function_call", "function_call_output"],
        )
    with patch("lite_agent.client.litellm.acompletion", mock2):
        await runner.run_continue_until_complete(
            includes=["assistant_message", "usage", "function_call", "function_call_output"],
        )


@pytest.mark.asyncio
async def test_agent_without_mock_data_fails():
    """Test that agent fails gracefully when mock data is missing."""
    # Use a non-existent directory
    mock = create_litellm_mock("tests/mocks/nonexistent/file.jsonl")

    with patch("lite_agent.client.litellm.acompletion", mock):
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
