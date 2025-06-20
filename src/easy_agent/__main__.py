import logging

from rich.logging import RichHandler

from easy_agent.agent import Agent
from easy_agent.loggers import logger
from easy_agent.runner import Runner

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()], format="%(message)s")
logger.setLevel(logging.DEBUG)


async def get_whether(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    return f"The temperature in {city} is 25 degrees Celsius."


async def main():
    agent = Agent(
        model="gpt-4.1-nano",
        name="Base Agent",
        instructions="You are a helpful assistant. Before using tools, tell the user what you are going to do.",
        tools=[get_whether, get_temperature],
    )
    runner = Runner(agent)
    async for chunk in runner.run_stream("What is the weather and temperature in San Francisco?"):
        print(chunk)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
