import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import display_messages
from lite_agent.client import LLMConfig, OpenAIClient
from lite_agent.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 25°C."


# Method 1: Using individual parameters
agent1 = Agent(
    model=OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=100,
        top_p=0.8,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=["END"],
    ),
    name="Weather Bot (Individual Params)",
    instructions="You are a weather assistant. Keep responses brief and factual.",
    tools=[get_weather],
)

# Method 2: Using LLMConfig object
llm_config = LLMConfig(
    temperature=0.8,
    max_tokens=200,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

agent2 = Agent(
    model=OpenAIClient(
        model="gpt-4o-mini",
        llm_config=llm_config,
    ),
    name="Weather Bot (LLMConfig)",
    instructions="You are a creative weather assistant. Add some personality to your responses.",
    tools=[get_weather],
)


async def main():
    # Test agent with conservative settings (low temperature, short responses)
    print("=== Testing Agent 1 (Conservative Settings) ===")
    runner1 = Runner(agent1)
    resp1 = runner1.run(
        "What's the weather like in Tokyo?",
        includes=["assistant_message"],
    )
    async for chunk in resp1:
        logger.info(chunk)
    display_messages(runner1.messages)

    print("\n" + "=" * 50 + "\n")

    # Test agent with creative settings (high temperature, longer responses)
    print("=== Testing Agent 2 (Creative Settings) ===")
    runner2 = Runner(agent2)
    resp2 = runner2.run(
        "What's the weather like in Tokyo?",
        includes=["assistant_message"],
    )
    async for chunk in resp2:
        logger.info(chunk)
    display_messages(runner2.messages)


if __name__ == "__main__":
    asyncio.run(main())
