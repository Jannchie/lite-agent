import asyncio
import logging

from rich import print  # noqa: A004
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


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(1)
    return f"The temperature in {city} is 25°C."


agent = Agent(
    model="gpt-4.1-nano",
    name="Weather Assistant",
    instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
    tools=[get_temperature],
)


async def main():
    runner = Runner(agent)
    resp = runner.run(
        "What is the temperature in New York?",
        includes=["usage", "assistant_message", "function_call", "function_call_output"],
        record_to="tests/mocks/basic/1.jsonl",
    )
    async for chunk in resp:
        logger.info(chunk)
    print(runner.messages)


if __name__ == "__main__":
    asyncio.run(main())
