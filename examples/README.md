# Examples Overview

The `examples/` directory contains runnable demonstrations for different Lite Agent features.  
All scripts can be executed with `uv run python <path-to-example>.py`, unless noted otherwise.

## Quickstart

- `basic.py` – Smallest end-to-end agent with a tool and context object.

## Categories

### basics/

- `basic_agent.py` – Call tools directly with the `Agent` API.
- `basic_model.py` – Minimal OpenAI client usage.
- `llm_config_demo.py` – Compare different configuration presets.
- `non_streaming.py` / `streaming_demo.py` – Switch between streaming modes.
- `reasoning_example.py` – Enable reasoning models.
- `response_api_example.py` / `responses.py` – Response API adapters.

### context/

- `context.py` – Pass a context container to tools.
- `context_modify.py` – Mutate history-aware context data.
- `context_todo.py` – Share mutable todo state across calls.
- `history_context_demo.py` – Auto-inject conversation history.
- `set_chat_history_example.py` – Restore and continue conversations.

### tools/

- `confirm_and_continue.py` – Require user confirmation before executing.
- `consolidate_history.py` – Test message-transfer helpers.
- `custom_termination.py` – Customize completion behavior.
- `message_transfer.py` – Register preprocessing callbacks.
- `stop_before_functions.py` / `stop_with_tool_call.py` – Control tool flow.
- `type_system_example.py` – Demonstrate typed tool schemas.

### workflows/

- `cancel_and_transfer_demo.py` – Transfer control between agents.
- `handoffs.py` – Parent/child agent delegation.
- `new_message_structure_demo.py` – Inspect message transformations.

### debug/

- `debug_non_streaming.py` – Inspect non-streaming chunk flow.
- `debug_with_logging.py` – Enable verbose logging for troubleshooting.
- `inspect_runner_modes.py` – Compare completion vs responses APIs.
- `simple_debug.py` – Lightweight sanity-check harness.

### structured/

- `structured_output_basic.py` – Quick structured output call.
- `structured_output_demo.py` – Stream structured output responses.
- `structured_output_responses_api.py` – Responses API variant.
- `structured_output_api_comparison.py` – Side-by-side API comparison.

### demos/

- `chat_display_demo.py` – Rich console rendering.
- `image.py` – Image generation walkthrough.
- `terminal.py` – Terminal-style chat loop.
- `translate_board.py` – Translation board agent with context and tools.
- `channels/` – Channel render helpers (e.g., Rich output).
- `knowledge/` – End-to-end knowledge base demo (`main.py`).
- `translate_app/` – Prompt-based translation workflow (`main.py`).

## Tips

- Set `OPENAI_API_KEY` (or another provider key) before running the scripts.
- Many examples stream output; add `--stream=false` or adjust `Runner(streaming=False)` when needed.
- Use the examples as starting points—copy them into your own workspace rather than editing in place.
