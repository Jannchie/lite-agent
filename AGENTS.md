# Repository Guidelines

## Project Structure & Module Organization

The core package lives in `src/lite_agent`, organized into focused subpackages such as `adapters`, `processors`, `stream_handlers`, and `utils` to keep agent logic modular. Tests reside under `tests/` and mirror the runtime layout to keep coverage tight. Example agents and runnable demos land in `examples/`, while `docs/` holds longer-form references. Automation helpers and one-off utilities belong in `scripts/`, and distribution artifacts are written to `dist/` during packaging runs.

## Build, Test, and Development Commands

Use `uv pip install -e .` (or `pip install -e .`) to bootstrap a development environment with local changes. Run `pytest` for the default suite, and append `--cov=src/lite_agent --cov-report=term-missing` when validating coverage before a pull request. Lint with `ruff check --fix src tests examples` to auto-resolve style issues, then follow up with `ruff check` to catch anything remaining. Invoke `pyright` to confirm static typing stays clean; configuration lives in `pyrightconfig.json`.

## Coding Style & Naming Conventions

Target Python 3.10+ syntax, keep indentation at four spaces, and respect the `line-length = 200` setting defined in `pyproject.toml`. Ruff enforces a broad selection of rules; rely on built-in types (`list`, `dict`, `tuple`) instead of legacy `typing` aliases. Every function, method, and public attribute needs explicit type annotations to align with the project’s "accurate and complete" typing standard. Modules and files use snake_case, classes use PascalCase, and tests follow the `test_<subject>.py` pattern.

## Testing Guidelines

Favor `pytest` for all checks and leverage `pytest-asyncio` for coroutine-heavy code paths. Keep test names descriptive, e.g., `test_runner_handles_streaming_response`, and group scenario helpers in local fixtures within the same module. When a change modifies agent behavior, include regression tests that assert streamed token order and tool-calling traces. Aim to maintain or raise coverage; highlight any intentional gaps in the pull request discussion.

## Commit & Pull Request Guidelines

Commits follow a Conventional Commit style (`feat(structured-output): ...`, `fix(examples): ...`), so include a scope that maps to the touched module. Each pull request should summarize the agent behavior change, list validation commands (`pytest`, `ruff check`, `pyright`), and reference related issues. Provide screenshots or terminal snippets when CLI output changes, and ensure new configuration flags are documented under `docs/` or `examples/` before requesting review.
