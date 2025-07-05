# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing

```bash
pytest                    # Run all tests
pytest tests/unit/        # Run only unit tests
pytest tests/integration/ # Run only integration tests
pytest --cov             # Run with coverage
```

### Linting and Formatting

```bash
ruff check               # Run linter
ruff format              # Format code
```

### Package Management

```bash
uv add lite-agent # Install from PyPI
uv add --dev lite-agent # Install dev package
```

## Project Architecture

LiteAgent is a lightweight AI agent framework built on top of LiteLLM. The core architecture consists of:

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

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test full agent workflows with mocked LLM responses
- **Performance tests**: Test memory usage and performance characteristics
- **Mock system**: JSONL-based conversation recording/playback for deterministic testing

The framework emphasizes simplicity and extensibility while maintaining full type safety and comprehensive streaming support.
