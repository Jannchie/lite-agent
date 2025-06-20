import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


async def get_whether(city: str) -> str:
    """Get the weather for a city."""
    await asyncio.sleep(1)
    return f"The weather in {city} is sunny with a few clouds."


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(1)
    return f"The temperature in {city} is 25°C."


async def main():
    agent = Agent(
        model="gpt-4.1",
        name="Weather Assistant",
        instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
        tools=[get_whether, get_temperature],
    )
    runner = Runner(agent)
    resp = await runner.run_until_complete("What is the weather in New York? And what is the temperature there?")
    for chunk in resp:
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
