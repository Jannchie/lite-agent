import asyncio
import logging

from funcall import Context
from openai import BaseModel
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import chat_summary_to_string, messages_to_string
from lite_agent.client import OpenAIClient
from lite_agent.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


class MyContext(BaseModel):
    d: str


ctx = Context(MyContext(d="example data"))


async def get_temperature(city: str, ctx: Context[MyContext]) -> str:
    """Get the temperature for a city."""
    print(ctx.value.d)
    return f"The temperature in {city} is 25°C."


agent = Agent(
    model=OpenAIClient(model="gpt-5-mini", reasoning={"effort": "minimal"}),
    name="Weather Assistant",
    instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses. You should call the get_temperature tool to get the temperature for a city.",
    tools=[get_temperature],
)


async def main():
    runner = Runner(agent)
    await runner.run_until_complete(
        "What is the temperature in New York?",
        includes=["usage", "assistant_message", "function_call", "function_call_output", "timing"],
        context=ctx,
    )
    messages = messages_to_string(runner.messages)
    summary = chat_summary_to_string(runner.messages)
    print(messages)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
