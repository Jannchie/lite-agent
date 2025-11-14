import json
import logging
import re
from functools import cache
from textwrap import dedent
from typing import TYPE_CHECKING, cast

from funcall import Context
from pydantic import BaseModel, Field

from lite_agent.client import LLMConfig, OpenAIClient

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LANGUAGE_LABELS = {
    "zh-Hans": "Simplified Chinese",
    "ja": "Japanese",
    "es": "Spanish",
}

TRANSLATION_MODEL = "gpt-4o-mini"

PLAN_STATUSES = {"pending", "in_progress", "completed"}

logger = logging.getLogger("lite_agent")


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

    def describe(self, project: "Project") -> str:
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


class PlanStep(BaseModel):
    step: str
    status: str


class PlanState(BaseModel):
    explanation: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)


class TranslationWorkspace(BaseModel):
    user_selection: SelectionState
    project: Project
    plan: PlanState = Field(default_factory=PlanState)


def _build_item_map(project: Project) -> dict[str, ProjectItem]:
    return {item.key: item for item in project.items}


def _format_meta(meta: dict[str, str]) -> str:
    if not meta:
        return "no meta"
    return json.dumps(meta, ensure_ascii=False)


def _workspace_from_context(ctx: Context[TranslationWorkspace]) -> TranslationWorkspace | None:
    return ctx.value


def _format_plan(plan: PlanState) -> str:
    if not plan.steps:
        return "Plan is currently empty."
    lines: list[str] = []
    if plan.explanation:
        lines.append(f"Note: {plan.explanation}")
    for index, entry in enumerate(plan.steps, start=1):
        lines.append(f"{index}. {entry.step} ({entry.status})")
    return "\n".join(lines)


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


@cache
def _get_translation_client() -> OpenAIClient:
    return OpenAIClient(
        model=TRANSLATION_MODEL,
        llm_config=LLMConfig(temperature=0.2, max_tokens=200),
    )


async def _translate_with_llm(text: str, source_language: str, target_language: str) -> str:
    if not text.strip():
        return text
    client = _get_translation_client()
    system_prompt = "You are a precise localization engine. Translate the provided text exactly once without commentary."
    user_prompt = dedent(
        f"""\
        Source language: {source_language}
        Target language: {target_language}
        Text:
        {text}
        """,
    ).strip()
    try:
        response = await client.completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            streaming=False,
        )
    except Exception as exc:
        logger.warning("Falling back to mock translation due to error: %s", exc)
        return _mock_translate(text, target_language)
    chat_completion = cast("ChatCompletion", response)
    message = chat_completion.choices[0].message.content if chat_completion.choices else None
    if not message:
        logger.warning("Empty translation response; using mock output.")
        return _mock_translate(text, target_language)
    return message.strip()


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


async def update_plan(
    ctx: Context[TranslationWorkspace],
    plan: list[PlanStep],
    explanation: str | None = None,
) -> str:
    """Update or clear the multi-step plan rendered to the user."""
    workspace = _workspace_from_context(ctx)
    if workspace is None:
        return "Workspace is missing."
    trimmed_steps: list[PlanStep] = []
    error_message: str | None = None
    initial_plan_empty = not workspace.plan.steps
    for entry in plan:
        step_text = entry.step.strip()
        status = entry.status.strip().lower()
        if not step_text:
            error_message = "Plan steps must include descriptive text."
            break
        if status not in PLAN_STATUSES:
            allowed = ", ".join(sorted(PLAN_STATUSES))
            error_message = f"Invalid status '{entry.status}'. Allowed statuses: {allowed}."
            break
        trimmed_steps.append(PlanStep(step=step_text, status=status))
    if error_message:
        return error_message
    if not trimmed_steps:
        workspace.plan.steps = []
        workspace.plan.explanation = explanation
        return "Plan cleared." if not explanation else f"Plan cleared. Note: {explanation}"
    if len(trimmed_steps) < 2:
        return "Plans must include at least two steps."
    if initial_plan_empty:
        for entry in trimmed_steps:
            entry.status = "pending"
    if sum(1 for entry in trimmed_steps if entry.status == "in_progress") > 1:
        return "Only one step can be in_progress at a time."
    workspace.plan.steps = trimmed_steps
    workspace.plan.explanation = explanation
    return f"Plan updated:\n{_format_plan(workspace.plan)}"


async def update_selection(
    ctx: Context[TranslationWorkspace],
    item_keys: list[str] | None = None,
    languages: list[str] | None = None,
    mode: str = "replace",
) -> str:
    """Update the user-side selection (replace, append, or clear)."""
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


async def translate_selection(
    ctx: Context[TranslationWorkspace],
    languages: list[str] | None = None,
    source_language: str | None = None,
) -> str:
    """
    Translate the current user selection using the provided languages.

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
            translated = await _translate_with_llm(source_text, source_lang, language)
            item.set_content(language, translated)
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


__all__ = [
    "PLAN_STATUSES",
    "LanguageRecord",
    "PlanState",
    "PlanStep",
    "Project",
    "ProjectItem",
    "SelectionState",
    "TranslationWorkspace",
    "find_items_by_content",
    "find_untranslated",
    "get_user_selection",
    "list_items",
    "set_content",
    "translate_selection",
    "update_plan",
    "update_selection",
]
