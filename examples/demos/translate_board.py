import asyncio
import logging
from datetime import datetime, timezone

from funcall import Context
from openai import BaseModel
from pydantic import Field
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import messages_to_string
from lite_agent.client import OpenAIClient
from lite_agent.context import HistoryContext
from lite_agent.runner import Runner


class TranslationItem(BaseModel):
    item_id: str
    source: str
    metadata: dict[str, str] | None = None


class TranslationRecord(BaseModel):
    translations: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    last_editor: str | None = None
    last_updated_at: datetime | None = None


class SelectionState(BaseModel):
    item_id: str | None = None
    field: str | None = None
    segment_index: int | None = None
    target_language: str | None = None


class TranslationWorkspace(BaseModel):
    items: list[TranslationItem]
    records: dict[str, TranslationRecord]
    selection: SelectionState
    source_language: str
    target_languages: list[str]


logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


def _resolve_target_language(workspace: TranslationWorkspace, override: str | None = None) -> str | None:
    if override:
        return override
    if workspace.selection.target_language:
        return workspace.selection.target_language
    if workspace.target_languages:
        return workspace.target_languages[0]
    return None


def _get_or_create_record(workspace: TranslationWorkspace, item_id: str) -> TranslationRecord:
    record = workspace.records.get(item_id)
    if record is None:
        record = TranslationRecord()
        workspace.records[item_id] = record
    return record


async def list_pending_items(ctx: Context[HistoryContext[TranslationWorkspace]]) -> str:
    workspace = ctx.value.data
    if workspace is None:
        return "Workspace is missing."
    pending_entries: list[str] = []
    for item in workspace.items:
        record = workspace.records.get(item.item_id)
        if record is None:
            pending_entries.append(f"{item.item_id} (missing all targets)")
            continue
        missing = [language for language in workspace.target_languages if not record.translations.get(language)]
        if missing:
            pending_entries.append(f"{item.item_id} (missing {', '.join(missing)})")
    if not pending_entries:
        return "All items currently have translations for every target language."
    return "\n".join(f"{index + 1}. {entry}" for index, entry in enumerate(pending_entries))


async def get_record_details(item_id: str, ctx: Context[HistoryContext[TranslationWorkspace]]) -> str:
    workspace = ctx.value.data
    if workspace is None:
        return "Workspace is missing."
    record = workspace.records.get(item_id)
    if record is None:
        return f"No record found for {item_id}."
    lines = [
        f"Source language: {workspace.source_language}",
        f"Target languages: {', '.join(workspace.target_languages)}",
    ]
    for language in workspace.target_languages:
        translation = record.translations.get(language)
        lines.append(f"{language}: {translation or '[pending]'}")
    if record.issues:
        lines.append("Issues:" + "\n - ".join(["", *record.issues]))
    if record.suggestions:
        lines.append("Suggestions:" + "\n - ".join(["", *record.suggestions]))
    return "\n".join(lines)


async def set_selection(
    item_id: str,
    field: str | None,
    ctx: Context[HistoryContext[TranslationWorkspace]],
    target_language: str | None = None,
) -> str:
    workspace = ctx.value.data
    if workspace is None:
        return "Workspace is missing."
    workspace.selection.item_id = item_id
    workspace.selection.field = field
    workspace.selection.segment_index = None
    resolved_language = _resolve_target_language(workspace, override=target_language)
    workspace.selection.target_language = resolved_language
    field_label = field or "translation"
    language_note = f" ({resolved_language})" if resolved_language else ""
    return f"Focused on item {item_id} field {field_label}{language_note}."


async def suggest_revision(item_id: str, ctx: Context[HistoryContext[TranslationWorkspace]]) -> str:
    workspace = ctx.value.data
    if workspace is None:
        return "Workspace is missing."
    record = _get_or_create_record(workspace, item_id)
    target_language = _resolve_target_language(workspace)
    if target_language is None:
        return "No target language available for suggestions."
    suggestion_text = "Consider clarifying the tone and ensure terminology consistency."
    entry = f"[{target_language}] {suggestion_text}"
    record.suggestions.append(entry)
    record.last_updated_at = datetime.now(timezone.utc)
    record.last_editor = "agent"
    return entry


async def apply_translation(item_id: str, new_text: str, ctx: Context[HistoryContext[TranslationWorkspace]]) -> str:
    workspace = ctx.value.data
    if workspace is None:
        return "Workspace is missing."
    record = _get_or_create_record(workspace, item_id)
    target_language = _resolve_target_language(workspace)
    if target_language is None:
        return "No target language selected. Use set_selection to specify one."
    record.translations[target_language] = new_text
    record.last_updated_at = datetime.now(timezone.utc)
    record.last_editor = "agent"
    return f"Updated translation for {item_id} ({target_language})."


async def flag_issue(item_id: str, issue_description: str, ctx: Context[HistoryContext[TranslationWorkspace]]) -> str:
    workspace = ctx.value.data
    if workspace is None:
        return "Workspace is missing."
    record = _get_or_create_record(workspace, item_id)
    target_language = _resolve_target_language(workspace)
    prefix = f"[{target_language}] " if target_language else ""
    record.issues.append(prefix + issue_description)
    record.last_updated_at = datetime.now(timezone.utc)
    record.last_editor = "agent"
    return f"Logged issue for {item_id}."


async def confirm_translation(item_id: str, ctx: Context[HistoryContext[TranslationWorkspace]]) -> str:
    workspace = ctx.value.data
    if workspace is None:
        return "Workspace is missing."
    record = _get_or_create_record(workspace, item_id)
    target_language = _resolve_target_language(workspace)
    if target_language is None:
        return "No target language selected. Use set_selection to specify one."
    translation = record.translations.get(target_language)
    if not translation:
        return f"No translation found for {item_id} in {target_language}."
    record.last_updated_at = datetime.now(timezone.utc)
    record.last_editor = "agent"
    return f"Translation for {item_id} ({target_language}) confirmed."


initial_workspace = TranslationWorkspace(
    source_language="en",
    target_languages=["zh-Hans"],
    items=[
        TranslationItem(item_id="headline", source="Breaking news from the conference."),
        TranslationItem(item_id="tagline", source="Innovate, Integrate, Inspire."),
    ],
    records={
        "headline": TranslationRecord(translations={"zh-Hans": "Conference breaking news."}),
        "tagline": TranslationRecord(),
    },
    selection=SelectionState(item_id="headline", field="translation", target_language="zh-Hans"),
)

agent = Agent(
    model=OpenAIClient(model="gpt-5-mini"),
    name="Translation Board Manager",
    instructions=(
        "You operate on a translation workspace. The source language is stored in context.data.source_language, "
        "and target languages appear in context.data.target_languages. Always check the current selection before "
        "editing and use set_selection to switch items or target languages. Use list_pending_items to see which entries "
        "lack translations, get_record_details for detailed status, suggest_revision before major edits, apply_translation "
        "to write updates, flag_issue when something needs attention, and confirm_translation to acknowledge that the "
        "current translation is acceptable. Treat empty translations as incomplete."
    ),
    tools=[
        list_pending_items,
        get_record_details,
        set_selection,
        suggest_revision,
        apply_translation,
        flag_issue,
        confirm_translation,
    ],
)


async def main() -> None:
    runner = Runner(agent)
    shared_context = Context(initial_workspace)
    await runner.run_until_complete("列出所有待翻译的条目，并检查正在关注的内容。", context=shared_context)
    await runner.run_until_complete("针对 headline 的译文提出改进建议，并更新内容。", context=shared_context)
    await runner.run_until_complete("把 tagline 设置为当前关注项，并执行初步翻译。", context=shared_context)
    await runner.run_until_complete("确认所有条目状态。", context=shared_context)
    print(messages_to_string(runner.messages))


if __name__ == "__main__":
    asyncio.run(main())
