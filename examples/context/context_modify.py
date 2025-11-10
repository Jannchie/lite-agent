import asyncio
import logging

from funcall import Context
from pydantic import BaseModel
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import display_messages
from lite_agent.context import HistoryContext
from lite_agent.runner import Runner


class UserCtx(BaseModel):
    name: str


logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


async def get_user_name(context: Context[HistoryContext[UserCtx]]) -> str:
    """Get the user name from the context."""
    if not context.value.data:
        msg = "User name must be specified in the context."
        raise ValueError(msg)
    return context.value.data.name


async def set_user_name(context: Context[HistoryContext[UserCtx]], new_user_name: str) -> str:
    """Set the user name in the context."""
    if not context.value.data:
        msg = "User name must be specified in the context."
        raise ValueError(msg)

    context.value.data.name = new_user_name
    return f"User name set to: {new_user_name}"


agent = Agent(
    model="gpt-4.1-nano",
    name="User Name Assistant",
    instructions="You can get and set the user name.",
    tools=[get_user_name, set_user_name],
)


async def main():
    runner = Runner(agent)
    user_ctx = Context(UserCtx(name="ATC"))
    resp = runner.run(
        "Set User name to TXA.",
        context=user_ctx,
    )
    async for chunk in resp:
        logger.info(chunk)
    await runner.run_until_complete("get the user's name", context=user_ctx)
    display_messages(runner.messages)


if __name__ == "__main__":
    asyncio.run(main())
