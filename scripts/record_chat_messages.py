import asyncio
import logging

from funcall.decorators import tool
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


@tool(require_confirmation=True)
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
        model="gpt-4.1-nano",
        name="Weather Assistant",
        instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
        tools=[get_whether, get_temperature],
    )
    runner = Runner(agent)
    resp = runner.run("What is the weather in New York? And what is the temperature there?")
    async for chunk in resp:
        logger.info(chunk)
    resp = runner.run_continue_stream()
    async for chunk in resp:
        logger.info(chunk)


if __name__ == "__main__":
    asyncio.run(main())
