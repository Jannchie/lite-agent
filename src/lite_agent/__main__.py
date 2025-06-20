import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.validation import Validator

from lite_agent.agent import Agent
from lite_agent.channels.rich_channel import RichChannel
from lite_agent.runner import Runner


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
    session = PromptSession()
    rich_channel = RichChannel()
    runner = Runner(agent)
    not_empty_validator = Validator.from_callable(
        lambda text: bool(text.strip()),
        error_message="Input cannot be empty.",
        move_cursor_to_end=True,
    )
    while True:
        try:
            user_input = await session.prompt_async(
                "👤 ",
                default="",
                complete_while_typing=True,
                validator=not_empty_validator,
                validate_while_typing=False,
            )
            if user_input.lower() in {"exit", "quit"}:
                break
            response = runner.run_stream(user_input)
            async for chunk in response:
                rich_channel.handle(chunk)

        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    asyncio.run(main())
