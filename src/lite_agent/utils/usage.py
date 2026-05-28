"""Helpers for normalizing provider token usage fields."""

from typing import Any


def extract_cached_input_tokens(usage: Any) -> int:  # noqa: ANN401
    """Return cached input tokens from OpenAI/LiteLLM usage payloads."""
    cached_tokens = _get_int_field(usage, ("cached_input_tokens", "cache_read_input_tokens"))
    if cached_tokens is not None:
        return cached_tokens

    for details_name in ("input_tokens_details", "prompt_tokens_details"):
        details = _get_field(usage, details_name)
        cached_tokens = _get_int_field(details, ("cached_tokens", "cached_input_tokens", "cache_read_input_tokens"))
        if cached_tokens is not None:
            return cached_tokens

    return 0


def _get_int_field(value: Any, names: tuple[str, ...]) -> int | None:  # noqa: ANN401
    for name in names:
        field_value = _get_field(value, name)
        if isinstance(field_value, int):
            return field_value
    return None


def _get_field(value: Any, name: str) -> Any:  # noqa: ANN401
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
