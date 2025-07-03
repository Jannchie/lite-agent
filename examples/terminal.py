import asyncio

from channels.rich_channel import RichChannel
from funcall.decorators import tool
from prompt_toolkit import PromptSession
from prompt_toolkit.validation import Validator

from lite_agent.agent import Agent
from lite_agent.runner import Runner


async def get_whether(city: str) -> str:
    """Get the weather for a city."""
    await asyncio.sleep(1)
    return f"The weather in {city} is sunny with a few clouds."


@tool(require_confirmation=True)
async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(1)
    return f"The temperature in {city} is 25°C."


async def main():
    agent = Agent(
        model="gpt-4.1",
        name="Weather Assistant",
        instructions="You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
        tools=[get_temperature, get_whether],
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
            response = runner.run(user_input)
            async for chunk in response:
                await rich_channel.handle(chunk)
            if await runner.has_require_confirm_tools():
                user_input = await session.prompt_async(
                    "❓ Confirm tool calls? (y/n) ",
                    default="y",
                    complete_while_typing=True,
                    validator=not_empty_validator,
                    validate_while_typing=False,
                )
                if user_input.lower() in {"y", "yes"}:
                    response = runner.run_continue_stream()
                    async for chunk in response:
                        await rich_channel.handle(chunk)
                else:
                    response = runner.run_continue_stream()
            rich_channel.new_turn()
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    asyncio.run(main())
