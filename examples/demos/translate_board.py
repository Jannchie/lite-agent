from __future__ import annotations

import asyncio
import json
import logging
import re

from funcall import Context
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.client import OpenAIClient
from lite_agent.runner import Runner

LANGUAGE_LABELS = {
    "zh-Hans": "Simplified Chinese",
    "ja": "Japanese",
    "es": "Spanish",
}


class LanguageRecord(BaseModel):
    language: str
    content: str


class ProjectItem(BaseModel):
    key: str
    meta: dict[str, str] = Field(default_factory=dict)
    records: list[LanguageRecord] = Field(default_factory=list)

    def content_for(self, language: str) -> str | None:
        for record in self.records:
            if record.language == language:
                return record.content
        return None

    def set_content(self, language: str, text: str) -> None:
        for record in self.records:
            if record.language == language:
                record.content = text
                return
        self.records.append(LanguageRecord(language=language, content=text))


class SelectionState(BaseModel):
    item_keys: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    def describe(self, project: Project) -> str:
        """Return a readable summary of the current selection."""
        if not self.item_keys:
            return "No items are currently selected."
        languages = self.languages or ([project.target_language] if project.target_language else [])
        if not languages:
            languages = [project.source_language]
        item_map = _build_item_map(project)
        lines: list[str] = []
        for key in self.item_keys:
            item = item_map.get(key)
            if item is None:
                continue
            meta_part = _format_meta(item.meta)
            lines.append(f"{key} ({meta_part})")
            for language in languages:
                text = item.content_for(language)
                cell = text if text else "[pending]"
                lines.append(f"  - {language}: {cell}")
        return "\n".join(lines) if lines else "Selection references unknown items."


class Project(BaseModel):
    source_language: str
    target_language: str
    items: list[ProjectItem] = Field(default_factory=list)


class TranslationWorkspace(BaseModel):
    user_selection: SelectionState
    project: Project


def _build_item_map(project: Project) -> dict[str, ProjectItem]:
    return {item.key: item for item in project.items}


def _format_meta(meta: dict[str, str]) -> str:
    if not meta:
        return "no meta"
    return json.dumps(meta, ensure_ascii=False)


def _workspace_from_context(ctx: Context[TranslationWorkspace]) -> TranslationWorkspace | None:
    return ctx.value


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _project_language_roster(project: Project) -> list[str]:
    roster: list[str] = []
    if project.source_language:
        roster.append(project.source_language)
    if project.target_language:
        roster.append(project.target_language)
    for item in project.items:
        roster.extend(record.language for record in item.records)
    return _unique(roster)


def _resolve_display_languages(project: Project, languages: list[str] | None) -> list[str]:
    if languages:
        return _unique(languages)
    return _project_language_roster(project)


def _resolve_target_languages(project: Project, languages: list[str] | None) -> list[str]:
    if languages:
        return [language for language in _unique(languages) if language]
    if project.target_language:
        return [project.target_language]
    return [language for language in _project_language_roster(project) if language != project.source_language]


def _mock_translate(text: str, target_language: str) -> str:
    label = LANGUAGE_LABELS.get(target_language, target_language)
    return f"{text} [{label}]"


async def list_items(ctx: Context[TranslationWorkspace], languages: list[str] | None = None) -> str:
    """List every item together with the requested languages."""
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    project = workspace.project
    languages_to_show = _resolve_display_languages(project, languages)
    lines: list[str] = []
    for item in sorted(project.items, key=lambda entry: entry.key):
        meta_part = _format_meta(item.meta)
        segments = [f"{item.key} ({meta_part})"]
        for language in languages_to_show:
            text = item.content_for(language)
            cell = text if text else "[pending]"
            segments.append(f"{language}: {cell}")
        lines.append(" | ".join(segments))
    return "\n".join(lines)


async def find_untranslated(ctx: Context[TranslationWorkspace], languages: list[str] | None = None) -> str:
    """Return which items are still missing a specific language."""
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    project = workspace.project
    languages_to_check = _resolve_target_languages(project, languages)
    if not languages_to_check:
        return "No target languages configured."
    lines: list[str] = []
    for language in languages_to_check:
        missing = [item.key for item in project.items if not item.content_for(language)]
        if missing:
            lines.append(f"{language}: {', '.join(missing)}")
        else:
            lines.append(f"{language}: all translated")
    return "\n".join(lines)


async def find_items_by_content(
    ctx: Context[TranslationWorkspace],
    query: str,
    *,
    use_regex: bool = False,
) -> str:
    """Search items whose content matches the given text (substring or regex) in any language."""
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    if not query:
        return "Please provide a non-empty query."
    project = workspace.project
    if use_regex:
        try:
            pattern = re.compile(query, flags=re.IGNORECASE)
        except re.error as exc:
            return f"Invalid regular expression: {exc}"

        def matches(text: str) -> bool:
            return bool(pattern.search(text))

    else:
        query_lower = query.lower()

        def matches(text: str) -> bool:
            return query_lower in text.lower()

    matched_items: list[str] = []
    for item in project.items:
        for record in item.records:
            if record.content and matches(record.content):
                matched_items.append(f"{item.key} ({record.language}): {record.content}")
                break
    if not matched_items:
        return f"No items contain '{query}'."
    return "\n".join(matched_items)


async def get_user_selection(ctx: Context[TranslationWorkspace]) -> str:
    """Inspect the selection coming from the UI."""
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    return workspace.user_selection.describe(workspace.project)


async def update_agent_selection(
    ctx: Context[TranslationWorkspace],
    item_keys: list[str] | None = None,
    languages: list[str] | None = None,
    mode: str = "replace",
) -> str:
    """Update the agent-side selection (replace, append, or clear)."""
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    selection = workspace.user_selection
    project = workspace.project
    normalized_mode = mode.lower()
    if normalized_mode == "clear":
        selection.item_keys = []
        selection.languages = []
        return "Selection cleared."
    item_map = _build_item_map(project)
    valid_keys: list[str] | None = None
    if item_keys is not None:
        valid_keys = [key for key in item_keys if key in item_map]
    normalized_languages: list[str] | None = None
    if languages is not None:
        normalized_languages = _unique(languages)
    if normalized_mode == "replace":
        if valid_keys is not None:
            selection.item_keys = valid_keys
        if normalized_languages is not None:
            selection.languages = normalized_languages
    elif normalized_mode == "append":
        if valid_keys is not None:
            for key in valid_keys:
                if key not in selection.item_keys:
                    selection.item_keys.append(key)
        if normalized_languages is not None:
            for language in normalized_languages:
                if language not in selection.languages:
                    selection.languages.append(language)
    else:
        return "Unsupported mode. Use replace, append, or clear."
    return f"Selection updated:\n{selection.describe(project)}"


async def translate_agent_selection(
    ctx: Context[TranslationWorkspace],
    languages: list[str] | None = None,
    source_language: str | None = None,
) -> str:
    """
    Translate the current selection using the provided languages.

    If languages are not provided, use the selection languages first, then fall back to the
    project's default target language.
    """
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    project = workspace.project
    selection = workspace.user_selection
    if not selection.item_keys:
        return "No items are currently selected."
    candidate_languages = languages or selection.languages
    resolved_targets = _resolve_target_languages(project, candidate_languages)
    target_languages = [language for language in resolved_targets if language != project.source_language]
    if not target_languages:
        return "No target languages specified."
    source_lang = source_language or project.source_language
    item_map = _build_item_map(project)
    applied: list[str] = []
    skipped_missing_source: list[str] = []
    for key in selection.item_keys:
        item = item_map.get(key)
        if item is None:
            continue
        source_text = item.content_for(source_lang)
        if not source_text:
            skipped_missing_source.append(f"{key} lacks {source_lang}")
            continue
        for language in target_languages:
            item.set_content(language, _mock_translate(source_text, language))
            applied.append(f"{key}:{language}")
    if not applied:
        return "No translations applied because sources were missing: " + ", ".join(skipped_missing_source) if skipped_missing_source else "No translations applied."
    message = f"Updated {len(applied)} cells: {', '.join(applied)}."
    if skipped_missing_source:
        message += f" Missing sources: {', '.join(skipped_missing_source)}."
    return message


async def set_content(
    ctx: Context[TranslationWorkspace],
    item_key: str,
    language: str,
    new_text: str,
) -> str:
    """Manually replace the content of a specific cell."""
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    project = workspace.project
    item_map = _build_item_map(project)
    item = item_map.get(item_key)
    if item is None:
        return f"Item {item_key} does not exist."
    item.set_content(language, new_text)
    return f"Updated {item_key} ({language})."


sample_items = [
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
            LanguageRecord(language="en", content="Set up your first workflow to unlock automation."),
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


initial_workspace = TranslationWorkspace(
    user_selection=SelectionState(item_keys=["landing.banner.title"], languages=["zh-Hans"]),
    project=Project(
        source_language="en",
        target_language="zh-Hans",
        items=sample_items,
    ),
)


logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


agent = Agent(
    model=OpenAIClient(model="gpt-5-mini"),
    name="Translation Board Controller",
    instructions=(
        "You manage a translation board whose context object always looks like "
        "{user_selection: {...}, project: {source_language, target_language, items}}. "
        "Each project item only exposes key, meta, and records[{language, content}]. "
        "The UI may contain an active selection that the user will not restate, so decide when to inspect it via get_user_selection before acting. "
        "Use update_agent_selection "
        "to change the agent-side selection (modes: replace, append, clear), list_items/find_untranslated/find_items_by_content "
        "to inspect the grid, translate_agent_selection to fill selected cells, "
        "and set_content when you must override a single cell manually."
    ),
    tools=[
        list_items,
        find_untranslated,
        find_items_by_content,
        get_user_selection,
        update_agent_selection,
        translate_agent_selection,
        set_content,
    ],
)


async def main() -> None:
    runner = Runner(agent)
    shared_context = Context(initial_workspace)

    await runner.run_until_complete("请概述当前翻译进度，也帮我看看界面里有没有需要立即关注的部分。", context=shared_context)

    initial_workspace.user_selection = SelectionState(
        item_keys=["landing.banner.title", "landing.banner.subtitle"],
        languages=["zh-Hans"],
    )
    await runner.run_until_complete("我想继续处理刚才正在编辑的中文内容，请直接把它们补齐。", context=shared_context)

    initial_workspace.user_selection = SelectionState()
    await runner.run_until_complete("接下来需要把缺少日语内容的条目全部翻译完成。", context=shared_context)

    initial_workspace.user_selection = SelectionState()
    await runner.run_until_complete("请找到文案里包含 workflow 的条目，并重新翻译它们的西语列覆盖旧内容。", context=shared_context)

    runner.display_message_history()


if __name__ == "__main__":
    asyncio.run(main())
