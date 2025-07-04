## v0.3.0

[v0.2.0...v0.3.0](https://github.com/Jannchie/lite-agent/compare/v0.2.0...v0.3.0)

### :rocket: Breaking Changes

- **rich-helpers**: rename render_chat_history to print_chat_history and update imports - By [Jannchie](mailto:jannchie@gmail.com) in [3db6721](https://github.com/Jannchie/lite-agent/commit/3db6721)

### :sparkles: Features

- **agent**: use jinja2 templates for instruction messages - By [Jannchie](mailto:jannchie@gmail.com) in [f5ccbd5](https://github.com/Jannchie/lite-agent/commit/f5ccbd5)
- **agent**: add completion_condition and task_done tool support - By [Jannchie](mailto:jannchie@gmail.com) in [28126b1](https://github.com/Jannchie/lite-agent/commit/28126b1)
- **response-api**: add response api format for text and image input - By [Jannchie](mailto:jannchie@gmail.com) in [4b44f91](https://github.com/Jannchie/lite-agent/commit/4b44f91)
- **rich-helpers**: add rich chat history renderer and chat summary utils - By [Jannchie](mailto:jannchie@gmail.com) in [5a570ce](https://github.com/Jannchie/lite-agent/commit/5a570ce)
- **runner**: add set_chat_history to support full chat history replay and agent tracking - By [Jannchie](mailto:jannchie@gmail.com) in [fd2ecdc](https://github.com/Jannchie/lite-agent/commit/fd2ecdc)
- **types**: enhance runner type system for flexible user_input - By [Jannchie](mailto:jannchie@gmail.com) in [3a2e9c5](https://github.com/Jannchie/lite-agent/commit/3a2e9c5)

### :adhesive_bandage: Fixes

- **examples**: check content key before modifying messages - By [Jannchie](mailto:jannchie@gmail.com) in [94b85a6](https://github.com/Jannchie/lite-agent/commit/94b85a6)

### :art: Refactors

- **agent**: abstract llm client and refactor model usage - By [Jannchie](mailto:jannchie@gmail.com) in [09c52c8](https://github.com/Jannchie/lite-agent/commit/09c52c8)
- **client**: move llm client classes to separate module - By [Jannchie](mailto:jannchie@gmail.com) in [a70d163](https://github.com/Jannchie/lite-agent/commit/a70d163)
- **rich-helpers-test**: rename render_chat_history to print_chat_history - By [Jannchie](mailto:jannchie@gmail.com) in [bf1ca8f](https://github.com/Jannchie/lite-agent/commit/bf1ca8f)

### :lipstick: Styles

- **agent**: inline error message formatting - By [Jannchie](mailto:jannchie@gmail.com) in [1453cc5](https://github.com/Jannchie/lite-agent/commit/1453cc5)
- **tests**: add type ignore hints to assertions - By [Jannchie](mailto:jannchie@gmail.com) in [8d2a19d](https://github.com/Jannchie/lite-agent/commit/8d2a19d)

### :memo: Documentation

- **set-chat-history**: remove set_chat_history feature documentation - By [Jannchie](mailto:jannchie@gmail.com) in [2413351](https://github.com/Jannchie/lite-agent/commit/2413351)

### :wrench: Chores

- **examples**: rename test_consolidate_history to consolidate_history - By [Jannchie](mailto:jannchie@gmail.com) in [0f5b39a](https://github.com/Jannchie/lite-agent/commit/0f5b39a)
- **tests**: remove legacy chat bubble and alignment tests - By [Jannchie](mailto:jannchie@gmail.com) in [46b23bc](https://github.com/Jannchie/lite-agent/commit/46b23bc)

## v0.2.0

[v0.1.0...v0.2.0](https://github.com/Jannchie/lite-agent/compare/v0.1.0...v0.2.0)

### :rocket: Breaking Changes

- **agent**: rename stream_async to completion - By [Jannchie](mailto:jannchie@gmail.com) in [dae25cc](https://github.com/Jannchie/lite-agent/commit/dae25cc)
- **messages**: add support for function call message types and responses format conversion - By [Jannchie](mailto:jannchie@gmail.com) in [9e71ccc](https://github.com/Jannchie/lite-agent/commit/9e71ccc)
- **project-structure**: rename package to lite-agent && update processor imports - By [Jannchie](mailto:jannchie@gmail.com) in [6e0747a](https://github.com/Jannchie/lite-agent/commit/6e0747a)
- **runner**: rename run_stream to run throughout codebase - By [Jannchie](mailto:jannchie@gmail.com) in [525865f](https://github.com/Jannchie/lite-agent/commit/525865f)
- **stream-handler**: rename litellm_raw to completion_raw - By [Jannchie](mailto:jannchie@gmail.com) in [9001f19](https://github.com/Jannchie/lite-agent/commit/9001f19)
- **types**: remove AgentToolCallMessage usage and definition - By [Jannchie](mailto:jannchie@gmail.com) in [fb3592e](https://github.com/Jannchie/lite-agent/commit/fb3592e)
- migrate chunk and message types to pydantic models && update runner and processors for model usage - By [Jannchie](mailto:jannchie@gmail.com) in [da0d6fc](https://github.com/Jannchie/lite-agent/commit/da0d6fc)

### :sparkles: Features

- **agent**: support dynamic handoff and parent transfer - By [Jannchie](mailto:jannchie@gmail.com) in [5019018](https://github.com/Jannchie/lite-agent/commit/5019018)
- **agent**: add async tool call handling - By [Jannchie](mailto:jannchie@gmail.com) in [7de9103](https://github.com/Jannchie/lite-agent/commit/7de9103)
- **context**: add context support for agent and runner - By [Jannchie](mailto:jannchie@gmail.com) in [e4885ee](https://github.com/Jannchie/lite-agent/commit/e4885ee)
- **handoffs**: support multiple agents and add weather tools - By [Jannchie](mailto:jannchie@gmail.com) in [b9440ea](https://github.com/Jannchie/lite-agent/commit/b9440ea)
- **handoffs**: add agent handoff and transfer functionality - By [Jannchie](mailto:jannchie@gmail.com) in [7655029](https://github.com/Jannchie/lite-agent/commit/7655029)
- **message-transfer**: add message_transfer callback support to agent and runner && provide consolidate_history_transfer utility && add tests and usage examples - By [Jannchie](mailto:jannchie@gmail.com) in [16d4788](https://github.com/Jannchie/lite-agent/commit/16d4788)
- **runner**: add record_to param to run_until_complete - By [Jannchie](mailto:jannchie@gmail.com) in [597bff6](https://github.com/Jannchie/lite-agent/commit/597bff6)
- **runner**: support chat stream recording and improve sequence typing - By [Jannchie](mailto:jannchie@gmail.com) in [5c1d609](https://github.com/Jannchie/lite-agent/commit/5c1d609)
- **runner**: add run_continue method and require confirm tool handling - By [Jannchie](mailto:jannchie@gmail.com) in [d130ebc](https://github.com/Jannchie/lite-agent/commit/d130ebc)
- **testing**: add integration and unit tests for agent and file recording - By [Jannchie](mailto:jannchie@gmail.com) in [e5a58ce](https://github.com/Jannchie/lite-agent/commit/e5a58ce)
- **tool**: add require_confirmation support for tool calls - By [Jannchie](mailto:panjianqi@preferred.jp) in [69625c8](https://github.com/Jannchie/lite-agent/commit/69625c8)
- **tool-calls**: add require_confirm chunk type and flow - By [Jannchie](mailto:panjianqi@preferred.jp) in [b2d1654](https://github.com/Jannchie/lite-agent/commit/b2d1654)
- **types**: add agent message and tool call types - By [Jannchie](mailto:jannchie@gmail.com) in [489e370](https://github.com/Jannchie/lite-agent/commit/489e370)

### :adhesive_bandage: Fixes

- **agent**: update tool target to completion - By [Jannchie](mailto:jannchie@gmail.com) in [6c8978c](https://github.com/Jannchie/lite-agent/commit/6c8978c)
- **examples**: simplify temper agent instructions - By [Jannchie](mailto:jannchie@gmail.com) in [6632092](https://github.com/Jannchie/lite-agent/commit/6632092)
- **loggers**: update logger name to lite_agent - By [Jannchie](mailto:jannchie@gmail.com) in [b4c59aa](https://github.com/Jannchie/lite-agent/commit/b4c59aa)
- **types**: change tool_calls from sequence to list - By [Jannchie](mailto:jannchie@gmail.com) in [334c9c0](https://github.com/Jannchie/lite-agent/commit/334c9c0)

### :art: Refactors

- **agent**: rename prepare_messages to prepare_completion_messages - By [Jannchie](mailto:jannchie@gmail.com) in [02eeed6](https://github.com/Jannchie/lite-agent/commit/02eeed6)
- **agent**: remove unused completions to responses converter - By [Jannchie](mailto:jannchie@gmail.com) in [f6432c6](https://github.com/Jannchie/lite-agent/commit/f6432c6)
- **chunk-handler**: update type annotations and chunk types - By [Jannchie](mailto:panjianqi@preferred.jp) in [f2b95c5](https://github.com/Jannchie/lite-agent/commit/f2b95c5)
- **chunk-handler**: extract helper functions for chunk processing - By [Jannchie](mailto:jannchie@gmail.com) in [083a164](https://github.com/Jannchie/lite-agent/commit/083a164)
- **litellm**: extract chunk processing logic to function - By [Jannchie](mailto:jannchie@gmail.com) in [f31e459](https://github.com/Jannchie/lite-agent/commit/f31e459)
- **litellm-stream**: decouple funcall from stream handler && adjust runner tool call logic - By [Jannchie](mailto:jannchie@gmail.com) in [2747e7e](https://github.com/Jannchie/lite-agent/commit/2747e7e)
- **rich-channel**: extract rich channel to separate module && clean main - By [Jannchie](mailto:jannchie@gmail.com) in [a8be74d](https://github.com/Jannchie/lite-agent/commit/a8be74d)
- **runner**: modularize runner methods and simplify message handling - By [Jannchie](mailto:jannchie@gmail.com) in [628a6b5](https://github.com/Jannchie/lite-agent/commit/628a6b5)
- **tests**: parametrize tool call type tests and clean up redundant cases - By [Jannchie](mailto:jannchie@gmail.com) in [0791582](https://github.com/Jannchie/lite-agent/commit/0791582)
- **types**: move chunk types to types.py && update type imports - By [Jannchie](mailto:panjianqi@preferred.jp) in [d2ebeec](https://github.com/Jannchie/lite-agent/commit/d2ebeec)

### :lipstick: Styles

- **tests**: improve formatting and consistency - By [Jannchie](mailto:jannchie@gmail.com) in [4342aa4](https://github.com/Jannchie/lite-agent/commit/4342aa4)

### :memo: Documentation

- add initial project readme - By [Jannchie](mailto:panjianqi@preferred.jp) in [f04ec7a](https://github.com/Jannchie/lite-agent/commit/f04ec7a)

### :wrench: Chores

- **ci**: switch to uv for dependency management and update to python 3.12 - By [Jannchie](mailto:panjianqi@preferred.jp) in [64d3085](https://github.com/Jannchie/lite-agent/commit/64d3085)
- **ci**: add github actions workflow - By [Jannchie](mailto:panjianqi@preferred.jp) in [051be1b](https://github.com/Jannchie/lite-agent/commit/051be1b)
- **deps**: update lock file - By [Jannchie](mailto:jannchie@gmail.com) in [9aa1e77](https://github.com/Jannchie/lite-agent/commit/9aa1e77)
- **metadata**: rename project and reformat keywords array - By [Jannchie](mailto:jannchie@gmail.com) in [2659c4d](https://github.com/Jannchie/lite-agent/commit/2659c4d)
- **metadata**: update ai topic classifier - By [Jannchie](mailto:jannchie@gmail.com) in [15d46ea](https://github.com/Jannchie/lite-agent/commit/15d46ea)
- **rename**: rename project name - By [Jannchie](mailto:jannchie@gmail.com) in [84077a8](https://github.com/Jannchie/lite-agent/commit/84077a8)

## v0.1.0

[5859f296f4aa3bf9156e560e5bbd98b3d8795b0d...v0.1.0](https://github.com/Jannchie/lite-agent/compare/5859f296f4aa3bf9156e560e5bbd98b3d8795b0d...v0.1.0)

### :sparkles: Features

- **chunk-handler**: add content-delta and tool-call-delta chunk types && update includes list in runner - By [Jannchie](mailto:jannchie@gmail.com) in [a3ba7d1](https://github.com/Jannchie/lite-agent/commit/a3ba7d1)
- **easy_agent**: add prompt toolkit dependency && update agent to use prompt session && improve chunk handling - By [Jannchie](mailto:jannchie@gmail.com) in [b408c14](https://github.com/Jannchie/lite-agent/commit/b408c14)

### :art: Refactors

- **main**: remove datetime imports and usage && add input validator - By [Jannchie](mailto:jannchie@gmail.com) in [eab45b3](https://github.com/Jannchie/lite-agent/commit/eab45b3)

### :wrench: Chores

- **import**: remove unused json tool import - By [Jannchie](mailto:jannchie@gmail.com) in [8de5975](https://github.com/Jannchie/lite-agent/commit/8de5975)
- **lock**: update lock file - By [Jannchie](mailto:jannchie@gmail.com) in [64880d3](https://github.com/Jannchie/lite-agent/commit/64880d3)
- **pyproject**: update version && improve description && expand keywords && add classifiers - By [Jannchie](mailto:jannchie@gmail.com) in [c3e8ee6](https://github.com/Jannchie/lite-agent/commit/c3e8ee6)
