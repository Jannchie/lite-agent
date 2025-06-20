import asyncio
import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.validation import Validator
from rich.console import Console
from rich.logging import RichHandler

from easy_agent.agent import Agent
from easy_agent.loggers import logger
from easy_agent.runner import Runner
from easy_agent.types import AgentChunk, ContentDeltaChunk

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()], format="%(message)s")
logger.setLevel(logging.INFO)


# --- Tool functions (English) ---
async def get_whether(city: str) -> str:
    """Get the weather for a city."""
    await asyncio.sleep(1)  # 模擬網路延遲
    return f"The weather in {city} is sunny with a few clouds."


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(1)  # 模擬網路延遲
    return f"The temperature in {city} is 25°C."


class RichChannel:
    def __init__(self) -> None:
        self.console = Console()
        self.map = {
            "final_message": self.handle_final_message,
            "tool_call": self.handle_tool_call,
            "tool_call_result": self.handle_tool_call_result,
            "tool_call_delta": self.handle_tool_call_delta,
            "content_delta": self.handle_content_delta,
            "usage": self.handle_usage,
        }
        self.new_turn = True

    def handle(self, chunk: AgentChunk):
        handler = self.map[chunk["type"]]
        handler(chunk)

    def handle_final_message(self, _chunk: AgentChunk):
        print()
        self.new_turn = True

    def handle_tool_call(self, chunk: AgentChunk):
        name = chunk.get("name", "<unknown>")
        arguments = chunk.get("arguments", "")
        self.console.print(f"🛠️  [green]{name}[/green]([yellow]{arguments}[/yellow])")

    def handle_tool_call_result(self, chunk: AgentChunk):
        name = chunk.get("name", "<unknown>")
        content = chunk.get("content", "")
        self.console.print(f"🛠️  [green]{name}[/green] → [yellow]{content}[/yellow]")

    def handle_tool_call_delta(self, chunk: AgentChunk): ...
    def handle_content_delta(self, chunk: ContentDeltaChunk):
        if self.new_turn:
            self.console.print("🤖 ", end="")
            self.new_turn = False
        print(chunk["delta"], end="", flush=True)

    def handle_usage(self, chunk: AgentChunk):
        if False:
            usage = chunk["usage"]
            self.console.print(f"In: {usage.prompt_tokens}, Out: {usage.completion_tokens}, Total: {usage.total_tokens}")


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
