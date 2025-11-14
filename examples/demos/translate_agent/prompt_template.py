from __future__ import annotations

from pathlib import Path

from jinja2 import Template

_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "translation_agent_prompt.md.j2"
_PROMPT_TEMPLATE = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))


def render_prompt(agent_description: str = "a localization agent", workspace_label: str = "agent") -> str:
    """Render the translation prompt with the desired labels."""
    return _PROMPT_TEMPLATE.render(
        agent_description=agent_description,
        workspace_label=workspace_label,
    )


TRANSLATE_AGENT_PROMPT = render_prompt()
