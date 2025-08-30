# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing

```bash
pytest                    # Run all tests
pytest tests/unit/        # Run only unit tests
pytest tests/integration/ # Run only integration tests
pytest tests/performance/ # Run performance tests
pytest --cov             # Run with coverage
pytest -v                # Verbose output
pytest -k "test_name"     # Run specific test
```

### Linting and Formatting

```bash
ruff check               # Run linter
ruff check --fix         # Run linter and auto-fix issues
ruff format              # Format code
pyright                  # Type checking (optional)
```

### Package Management

```bash
uv install              # Install all dependencies
uv add <package>         # Add a new dependency
uv add --dev <package>   # Add a development dependency
uv sync                  # Sync dependencies from lock file
uv run <command>         # Run command in project environment
```

### Running Examples

```bash
uv run python examples/basic.py                    # Simple agent with tool calling
uv run python examples/handoffs.py                 # Agent-to-agent transfers
uv run python examples/chat_display_demo.py        # Rich console output
uv run python examples/context.py                  # Context passing to tools
uv run python examples/terminal.py                 # Terminal-based interaction
uv run python examples/translate/main.py           # Translation agent example
uv run python examples/streaming_demo.py           # Streaming responses demo
uv run python examples/response_api_example.py     # Response API format demo
uv run python scripts/record_chat_messages.py      # Record conversations for testing
```

## Project Architecture

LiteAgent is a lightweight AI agent framework designed for flexibility with any LLM provider. The core architecture consists of:

### Core Components

**Agent (`src/lite_agent/agent.py`)**

- Central agent class that manages LLM interactions, tool calls, and message handling
- Supports tool registration via `funcall` library for type-safe function calling
- Handles agent handoffs (parent-child relationships) for complex workflows
- Manages completion conditions ("stop" vs "call" for different termination behaviors)
- Converts between OpenAI's Response API and Completion API message formats

**Runner (`src/lite_agent/runner.py`)**

- Orchestrates agent execution with streaming support
- Manages conversation flow and message history
- Handles tool call execution and agent transfers
- Supports continuation from previous states and chat history management
- Provides both streaming and batch execution modes

**Type System (`src/lite_agent/types/`)**

- Comprehensive Pydantic models for all message types and chunks
- Supports both Response API and Completion API formats
- Type-safe definitions for tool calls, chunks, and messages

### Key Features

**Tool Integration**

- Uses `funcall` library for automatic tool schema generation from Python functions
- Supports basic types, Pydantic models, and dataclasses as parameters
- Dynamic tool registration for agent handoffs and control flow

**Agent Handoffs**

- Parent-child agent relationships for complex task delegation
- Automatic `transfer_to_agent` and `transfer_to_parent` tool registration
- Chat history tracking across agent transitions

**Message Processing**

- Bidirectional conversion between OpenAI API formats
- Streaming chunk processing with configurable output filtering
- Message transfer callbacks for preprocessing

**Completion Modes**

- `"stop"`: Traditional completion until model decides to stop
- `"call"`: Completion until specific tool (`wait_for_user`) is called

### Examples Directory Structure

Examples demonstrate various usage patterns:

- `basic.py`: Simple agent with tool calling
- `handoffs.py`: Agent-to-agent transfers
- `context.py`: Context passing to tools
- `chat_display_demo.py`: Rich console output formatting
- `channels/`: Channel-based communication patterns

### Testing Architecture

- **Unit tests**: Test individual components in isolation (`tests/unit/`)
- **Integration tests**: Test full agent workflows with mocked LLM responses (`tests/integration/`)
- **Performance tests**: Test memory usage and performance characteristics (`tests/performance/`)
- **Mock system**: JSONL-based conversation recording/playback for deterministic testing (`tests/mocks/`)

**Recording Mock Conversations**: Use `scripts/record_chat_messages.py` to capture real LLM interactions for test scenarios

### API Compatibility

The framework supports two OpenAI API modes:

- **Response API** (default): Modern structured response format
- **Completion API**: Legacy completion format for backward compatibility

Set via `Runner(agent, api="completion")` or `Runner(agent, api="responses")`.

### Message Types and Streaming

The framework provides rich message types supporting both text and structured content:

- **Text messages**: Simple string content for basic interactions
- **Tool calls**: Structured function calls with parameters and results
- **Agent transfers**: Built-in support for handoffs between specialized agents
- **Rich content**: Support for complex message structures via Pydantic models
- **Streaming chunks**: Real-time processing of LLM responses with granular event types

### Development Notes

- Project uses strict ruff linting with `select = ["ALL"]` and specific ignores
- All functions require full type annotations
- Uses `uv` for package management and dependency resolution  
- Mock conversations stored in `tests/mocks/` as JSONL files for reproducible testing
- Examples in `examples/` directory demonstrate various usage patterns
- Template system uses Jinja2 for dynamic instruction generation (`src/lite_agent/templates/`)
- Does NOT directly depend on `litellm` - works with any compatible LLM client via `BaseLLMClient` interface
- Chat display functionality uses `rich` library for formatted console output
- Uses `pyright` for type checking with custom configuration excluding examples and temp directories

The framework emphasizes simplicity and extensibility while maintaining full type safety and comprehensive streaming support.
