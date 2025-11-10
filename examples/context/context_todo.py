import asyncio
import logging

from funcall import Context
from openai import BaseModel
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import messages_to_string
from lite_agent.client import OpenAIClient
from lite_agent.runner import Runner


class TodoItem(BaseModel):
    title: str
    done: bool = False


class TodoBoard(BaseModel):
    items: list[TodoItem] = []


todo_board = TodoBoard(
    items=[
        TodoItem(title="Write unit tests"),
        TodoItem(title="Review PR #42"),
    ],
)


async def list_todos(ctx: Context[TodoBoard]) -> str:
    """Return a human-readable list of outstanding todo items."""
    ctx_data = ctx.value
    if not ctx_data.items:
        return "Your todo list is empty."
    lines = []
    for index, item in enumerate(ctx_data.items, start=1):
        status = "✅" if item.done else "⬜"
        lines.append(f"{index}. {status} {item.title}")
    return "\n".join(lines)


async def add_todo(ctx: Context[TodoBoard], title: str) -> str:
    """Append a new todo item to the shared board."""
    ctx_data = ctx.value
    ctx_data.items.append(TodoItem(title=title))
    return f"Added todo: {title}"


async def complete_todo(ctx: Context[TodoBoard], index: int) -> str:
    """Mark a todo item as completed."""
    ctx_data = ctx.value
    if index < 1 or index > len(ctx_data.items):
        return "Invalid todo index."
    ctx_data.items[index - 1].done = True
    return f"Marked todo #{index} as complete."


logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


agent = Agent(
    model=OpenAIClient(model="gpt-5-mini", reasoning={"effort": "minimal"}),
    name="Todo Manager",
    instructions=(
        "Assist the user with their todo list. "
        "Use list_todos to inspect current items, add_todo to create new entries, "
        "and complete_todo to mark tasks as done."
    ),
    tools=[list_todos, add_todo, complete_todo],
)


async def main() -> None:
    runner = Runner(agent)
    await runner.run_until_complete("What is on my todo list?", context=todo_board)
    await runner.run_until_complete("Add a todo for preparing the demo.", context=todo_board)
    await runner.run_until_complete("Mark the second todo as done.", context=todo_board)
    await runner.run_until_complete("Show me the updated todos.", context=todo_board)
    print(messages_to_string(runner.messages))


if __name__ == "__main__":
    asyncio.run(main())
