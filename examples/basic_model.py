import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import display_messages
from lite_agent.client import LiteLLMClient
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
    model=LiteLLMClient(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=150,
        top_p=0.9,
    ),
    name="Weather Assistant",
    instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
    tools=[get_temperature],
)


async def main():
    runner = Runner(agent)
    resp = runner.run(
        "What is the temperature in New York?",
        includes=["usage", "assistant_message", "function_call", "function_call_output", "timing"],
    )
    async for chunk in resp:
        logger.info(chunk)
    display_messages(runner.messages)
    print(runner.messages)


if __name__ == "__main__":
    asyncio.run(main())
