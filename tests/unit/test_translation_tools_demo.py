from __future__ import annotations

import pytest
from funcall import Context

from examples.demos.translate_agent.translation_tools import (
    LanguageRecord,
    Project,
    ProjectItem,
    SelectionState,
    TranslationWorkspace,
    update_selection,
)


def _build_workspace() -> TranslationWorkspace:
    items = [
        ProjectItem(
            key="landing.hero.title",
            records=[
                LanguageRecord(language="en", content="Heading"),
                LanguageRecord(language="ja", content=""),
                LanguageRecord(language="es", content="Encabezado"),
            ],
        ),
        ProjectItem(
            key="landing.hero.subtitle",
            records=[
                LanguageRecord(language="en", content="Subheading"),
                LanguageRecord(language="ja", content="サブタイトル"),
                LanguageRecord(language="es", content=""),
            ],
        ),
        ProjectItem(
            key="dashboard.card.helper",
            records=[
                LanguageRecord(language="en", content="Helper"),
                LanguageRecord(language="ja", content=""),
                LanguageRecord(language="es", content=""),
            ],
        ),
    ]
    project = Project(source_language="en", target_language="ja", items=items)
    return TranslationWorkspace(user_selection=SelectionState(), project=project)


@pytest.mark.asyncio
async def test_update_selection_accepts_wildcard_patterns() -> None:
    workspace = _build_workspace()
    ctx = Context(workspace)

    result = await update_selection(ctx, key_patterns=["landing.*"], mode="replace")

    assert "landing.hero.title" in workspace.user_selection.item_keys
    assert "landing.hero.subtitle" in workspace.user_selection.item_keys
    assert "dashboard.card.helper" not in workspace.user_selection.item_keys
    assert "landing.hero.title" in result


@pytest.mark.asyncio
async def test_update_selection_selects_missing_languages_any() -> None:
    workspace = _build_workspace()
    ctx = Context(workspace)

    await update_selection(ctx, missing_languages=["ja"], mode="replace")

    assert workspace.user_selection.item_keys == ["landing.hero.title", "dashboard.card.helper"]


@pytest.mark.asyncio
async def test_update_selection_requires_all_missing_languages() -> None:
    workspace = _build_workspace()
    ctx = Context(workspace)

    await update_selection(
        ctx,
        missing_languages=["ja", "es"],
        missing_languages_mode="all",
    )

    assert workspace.user_selection.item_keys == ["dashboard.card.helper"]
