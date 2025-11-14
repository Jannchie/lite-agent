from __future__ import annotations

import argparse
import asyncio
import logging

from funcall import Context
from prompt_template import TRANSLATE_AGENT_PROMPT
from prompt_toolkit import PromptSession
from prompt_toolkit.validation import Validator
from rich.logging import RichHandler
from translation_tools import (
    LanguageRecord,
    Project,
    ProjectItem,
    SelectionState,
    TranslationWorkspace,
    find_items_by_content,
    find_untranslated,
    get_user_selection,
    list_items,
    set_content,
    translate_selection,
    update_plan,
    update_selection,
)

from examples.demos.channels.rich_channel import RichChannel
from lite_agent.agent import Agent
from lite_agent.client import OpenAIClient
from lite_agent.runner import Runner

SAMPLE_ITEMS = [
    ProjectItem(
        key="landing.banner.title",
        meta={"module": "landing", "category": "marketing"},
        records=[
            LanguageRecord(language="en", content="Bold ideas for modern teams"),
            LanguageRecord(language="zh-Hans", content=""),
            LanguageRecord(language="ja", content=""),
            LanguageRecord(language="es", content=""),
        ],
    ),
    ProjectItem(
        key="landing.banner.subtitle",
        meta={"module": "landing", "category": "marketing"},
        records=[
            LanguageRecord(language="en", content="Product updates delivered live from the summit."),
            LanguageRecord(language="zh-Hans", content="来自峰会的产品更新直播。"),
            LanguageRecord(language="ja", content=""),
            LanguageRecord(language="es", content=""),
        ],
    ),
    ProjectItem(
        key="dashboard.empty_state.title",
        meta={"module": "dashboard", "category": "empty_state"},
        records=[
            LanguageRecord(language="en", content="There are no workflows yet."),
            LanguageRecord(language="zh-Hans", content="暂无工作流。"),
            LanguageRecord(language="ja", content=""),
            LanguageRecord(language="es", content=""),
        ],
    ),
    ProjectItem(
        key="dashboard.empty_state.helper",
        meta={"module": "dashboard", "category": "empty_state"},
        records=[
            LanguageRecord(
                language="en",
                content="Set up your first workflow to unlock automation.",
            ),
            LanguageRecord(language="zh-Hans", content=""),
            LanguageRecord(language="ja", content=""),
            LanguageRecord(language="es", content="Configura tu primer flujo para activar la automatización."),
        ],
    ),
    ProjectItem(
        key="onboarding.checklist.title",
        meta={"module": "onboarding", "category": "onboarding"},
        records=[
            LanguageRecord(language="en", content="Complete the rollout checklist"),
            LanguageRecord(language="zh-Hans", content="完成上线清单"),
            LanguageRecord(language="ja", content=""),
            LanguageRecord(language="es", content=""),
        ],
    ),
    ProjectItem(
        key="automation.workspace.blurb",
        meta={"module": "automation", "category": "product"},
        records=[
            LanguageRecord(language="en", content="Automation keeps every workflow in sync."),
            LanguageRecord(language="zh-Hans", content="自动化保持每个工作流同步。"),
            LanguageRecord(language="ja", content=""),
            LanguageRecord(language="es", content="La automatización mantiene sincronizado cada flujo de trabajo."),
        ],
    ),
]


def _clone_project_items(items: list[ProjectItem]) -> list[ProjectItem]:
    return [item.model_copy(deep=True) for item in items]


def build_workspace() -> TranslationWorkspace:
    return TranslationWorkspace(
        user_selection=SelectionState(item_keys=["landing.banner.title"], languages=["zh-Hans"]),
        project=Project(
            source_language="en",
            target_language="zh-Hans",
            items=_clone_project_items(SAMPLE_ITEMS),
        ),
    )


initial_workspace = build_workspace()


logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


agent = Agent(
    model=OpenAIClient(model="gpt-5-mini"),
    name="Translation Agent Controller",
    instructions=TRANSLATE_AGENT_PROMPT,
    tools=[
        list_items,
        find_untranslated,
        find_items_by_content,
        get_user_selection,
        update_plan,
        update_selection,
        translate_selection,
        set_content,
    ],
)


async def scripted_walkthrough() -> None:
    runner = Runner(agent)
    shared_context = Context(initial_workspace)

    await runner.run_until_complete(
        "Please summarize the current translation progress and call out anything urgent on the agent.",
        context=shared_context,
    )

    initial_workspace.user_selection = SelectionState(
        item_keys=["landing.banner.title", "landing.banner.subtitle"],
        languages=["zh-Hans"],
    )
    await runner.run_until_complete(
        "I want to continue filling in the Chinese entries I was editing just now—please finish them.",
        context=shared_context,
    )

    initial_workspace.user_selection = SelectionState()
    await runner.run_until_complete(
        "Next, translate every item that is still missing Japanese content.",
        context=shared_context,
    )

    initial_workspace.user_selection = SelectionState()
    await runner.run_until_complete(
        "Find the entries containing the word 'workflow' and refresh their Spanish columns with new translations.",
        context=shared_context,
    )

    runner.display_message_history()


async def run_interactive_cli() -> None:
    """Launch an interactive CLI session for the translation agent."""
    workspace = build_workspace()
    shared_context = Context(workspace)
    runner = Runner(agent)
    rich_channel = RichChannel()
    session = PromptSession()
    not_empty_validator = Validator.from_callable(
        lambda text: bool(text.strip()),
        error_message="Input cannot be empty.",
        move_cursor_to_end=True,
    )
    print("Translation agent demo. Type 'exit' to quit.")
    while True:
        try:
            user_input = await session.prompt_async(
                "> ",
                default="",
                complete_while_typing=True,
                validator=not_empty_validator,
                validate_while_typing=False,
            )
        except (EOFError, KeyboardInterrupt):
            print("\nSession cancelled.")
            break
        normalized_input = user_input.strip()
        if not normalized_input:
            continue
        if normalized_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        response = runner.run(normalized_input, context=shared_context)
        async for chunk in response:
            await rich_channel.handle(chunk)
        if await runner.has_require_confirm_tools():
            while True:
                confirm = await session.prompt_async(
                    "❓ Confirm tool calls? (y/n) ",
                    default="y",
                    complete_while_typing=True,
                    validator=not_empty_validator,
                    validate_while_typing=False,
                )
                answer = confirm.strip().lower()
                if answer in {"y", "yes"}:
                    response = runner.run(None, context=shared_context)
                    async for chunk in response:
                        await rich_channel.handle(chunk)
                    break
                if answer in {"n", "no"}:
                    response = runner.run(None, context=shared_context)
                    async for chunk in response:
                        await rich_channel.handle(chunk)
                    break
                print("Please answer y or n.")
        rich_channel.new_turn()
    print("\nConversation transcript:")
    runner.display_message_history()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translation agent demo runner.")
    parser.add_argument(
        "--mode",
        choices=("scripted", "interactive"),
        default="scripted",
        help="Choose the scripted walkthrough or interactive CLI.",
    )
    parser.add_argument(
        "--log-level",
        choices=tuple(LOG_LEVELS),
        default="warning",
        help="Set logging verbosity for lite_agent internals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.setLevel(LOG_LEVELS[args.log_level])
    if args.mode == "interactive":
        asyncio.run(run_interactive_cli())
    else:
        asyncio.run(scripted_walkthrough())


if __name__ == "__main__":
    main()
