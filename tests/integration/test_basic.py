from unittest.mock import patch

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from tests.utils.mock_litellm import create_litellm_mock


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    return f"The temperature in {city} is 25°C."


agent = Agent(
    model="gpt-4.1-nano",
    name="Weather Assistant",
    instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
    tools=[get_temperature],
)
runner = Runner(agent, api="completion")


@pytest.mark.asyncio
async def test_basic():
    mock = create_litellm_mock("tests/mocks/basic/1.jsonl")
    with patch("lite_agent.client.litellm.acompletion", mock):
        resp = runner.run(
            "What is the temperature in New York?",
        )
        async for chunk in resp:
            print(chunk)
