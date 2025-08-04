import asyncio
import logging

from anyio import Path
from jinja2 import Environment, FileSystemLoader
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

translation_system = "translation_system.md.j2"
env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "prompts"),
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=True,
)
template = env.get_template(translation_system)
agent = Agent(
    model="gpt-4.1-nano",
    name="Translate Assistant",
    instructions=template.render(target_language="Chinese"),
)


async def main():
    runner = Runner(agent)
    resp = runner.run(
        """突然お邪魔してごめんなさい\\nびっくりしましたよね......""",
        includes=["usage", "assistant_message", "function_call", "function_call_output", "timing"],
    )
    async for chunk in resp:
        logger.info(chunk)
    display_messages(runner.messages)


if __name__ == "__main__":
    asyncio.run(main())
