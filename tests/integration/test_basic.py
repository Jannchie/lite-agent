from unittest.mock import Mock, patch

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from tests.utils.mock_openai import create_chat_completion_stream_mock


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    return f"The temperature in {city} is 25°C."



@pytest.mark.asyncio
async def test_basic():
    completion_mock = create_chat_completion_stream_mock("tests/mocks/basic/1.jsonl")

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = completion_mock
    mock_client.responses = Mock()

    with patch("lite_agent.client.AsyncOpenAI", return_value=mock_client):
        agent = Agent(
            model="gpt-4.1-nano",
            name="Weather Assistant",
            instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
            tools=[get_temperature],
        )
        runner = Runner(agent, api="completion")

        resp = runner.run("What is the temperature in New York?")
        async for chunk in resp:
            print(chunk)
