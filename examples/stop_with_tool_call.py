import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import display_messages
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
    return f"The temperature in {city} is 25°C."


agent = Agent(
    model="gpt-4.1",
    name="Weather Assistant",
    instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
    tools=[get_temperature],
    completion_condition="call",  # Set completion condition to "call"
)


async def main():
    runner = Runner(agent)
    resp = runner.run(
        "What is the temperature and whether in New York?",
        includes=["assistant_message", "usage", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        logger.info(chunk)
    display_messages(runner.messages)


if __name__ == "__main__":
    asyncio.run(main())
