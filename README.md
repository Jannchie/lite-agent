# LiteAgent

[![codecov](https://codecov.io/gh/Jannchie/lite-agent/graph/badge.svg?token=SJW89Z1VAZ)](https://codecov.io/gh/Jannchie/lite-agent)

## Introduction

LiteAgent is an easy-to-learn, lightweight, and extensible AI agent framework built on top of [LiteLLM](https://github.com/BerriAI/litellm). It is designed as a minimal yet practical implementation for quickly building intelligent assistants and chatbots with robust tool-calling capabilities. The codebase is intentionally simple, making it ideal for learning, extension, and rapid prototyping.

**Key Advantages:**

- **Minimal and approachable:** The simplest agent implementation for fast learning and hacking.
- **Accurate and complete type hints:** All function signatures are fully type-hinted and never faked, ensuring reliable developer experience and static analysis.
- **Flexible parameter definition:** Supports defining tool function parameters using basic types, Pydantic models, or Python dataclasses—even in combination.
- **Streaming responses:** Seamless support for LiteLLM streaming output.
- **Custom tool functions:** Easily integrate your own Python functions (e.g., weather, temperature queries).
- **Rich type annotations, Pydantic-based.**
- **Easy to extend and test.**

## Installation

You can install LiteAgent directly from PyPI:

```bash
pip install lite-agent
```

Or use [uv](https://github.com/astral-sh/uv):

```bash
uv pip install lite-agent
```

If you want to install from source for development:

```bash
uv pip install -e .
# or
pip install -e .
```

## Quick Start

### Code Example

See `examples/basic.py`:

```python
import asyncio
from lite_agent.agent import Agent
from lite_agent.runner import Runner

async def get_whether(city: str) -> str:
    await asyncio.sleep(1)
    return f"The weather in {city} is sunny with a few clouds."

async def main():
    agent = Agent(
        model="gpt-4.1",
        name="Weather Assistant",
        instructions="You are a helpful weather assistant.",
        tools=[get_whether],
    )
    runner = Runner(agent)
    resp = await runner.run_until_complete("What's the weather in New York?")
    for chunk in resp:
        print(chunk)

if __name__ == "__main__":
    asyncio.run(main())
```

See `pyproject.toml` for details.

## Testing

```bash
pytest
```

## License

MIT License

## Author

Jianqi Pan ([jannchie@gmail.com](mailto:jannchie@gmail.com))
