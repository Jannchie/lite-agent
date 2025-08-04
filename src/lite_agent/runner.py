import json
from collections.abc import AsyncGenerator, Sequence
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from lite_agent.agent import Agent
from lite_agent.loggers import logger
from lite_agent.types import (
    AgentChunk,
    AgentChunkType,
    AssistantMessageContent,
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    FlexibleRunnerMessage,
    MessageDict,
    MessageUsage,
    NewAssistantMessage,
    NewMessage,
    NewSystemMessage,
    # New structured message types
    NewUserMessage,
    ToolCall,
    ToolCallFunction,
    UserImageContent,
    UserInput,
    UserMessageContent,
    UserTextContent,
)

DEFAULT_INCLUDES: tuple[AgentChunkType, ...] = (
    "completion_raw",
    "usage",
    "function_call",
    "function_call_output",
    "content_delta",
    "function_call_delta",
    "assistant_message",
)


class Runner:
    def __init__(self, agent: Agent, api: Literal["completion", "responses"] = "responses") -> None:
        self.agent = agent
        self.messages: list[NewMessage] = []
        self.api = api
        self._current_assistant_message: NewAssistantMessage | None = None

    @property
    def legacy_messages(self) -> list[NewMessage]:
        """Return messages in new format (legacy_messages is now an alias)."""
        return self.messages

    def _start_assistant_message(self, content: str = "", meta: AssistantMessageMeta | None = None) -> None:
        """Start a new assistant message."""
        if meta is None:
            meta = AssistantMessageMeta()

        # Always add text content, even if empty (we can update it later)
        assistant_content_items: list[AssistantMessageContent] = [AssistantTextContent(text=content)]
        self._current_assistant_message = NewAssistantMessage(
            content=assistant_content_items,
            meta=meta,
        )

    def _add_to_current_assistant_message(self, content_item: AssistantTextContent | AssistantToolCall | AssistantToolCallResult) -> None:
        """Add content to the current assistant message."""
        if self._current_assistant_message is None:
            self._start_assistant_message()

        if self._current_assistant_message is not None:
            self._current_assistant_message.content.append(content_item)

    def _add_text_content_to_current_assistant_message(self, delta: str) -> None:
        """Add text delta to the current assistant message's text content."""
        if self._current_assistant_message is None:
            self._start_assistant_message()

        if self._current_assistant_message is not None:
            # Find the first text content item and append the delta
            for content_item in self._current_assistant_message.content:
                if content_item.type == "text":
                    content_item.text += delta
                    return
            # If no text content found, add new text content
            new_content = AssistantTextContent(text=delta)
            self._current_assistant_message.content.append(new_content)

    def _finalize_assistant_message(self) -> None:
        """Finalize the current assistant message and add it to messages."""
        if self._current_assistant_message is not None:
            self.messages.append(self._current_assistant_message)
            self._current_assistant_message = None

    def _add_tool_call_result(self, call_id: str, output: str, execution_time_ms: int | None = None) -> None:
        """Add a tool call result to the last assistant message, or create a new one if needed."""
        result = AssistantToolCallResult(
            call_id=call_id,
            output=output,
            execution_time_ms=execution_time_ms,
        )

        if self.messages and isinstance(self.messages[-1], NewAssistantMessage):
            # Add to existing assistant message
            self.messages[-1].content.append(result)
        else:
            # Create new assistant message with just the tool result
            assistant_message = NewAssistantMessage(content=[result])
            self.messages.append(assistant_message)

    def _normalize_includes(self, includes: Sequence[AgentChunkType] | None) -> Sequence[AgentChunkType]:
        """Normalize includes parameter to default if None."""
        return includes if includes is not None else DEFAULT_INCLUDES

    def _normalize_record_path(self, record_to: PathLike | str | None) -> Path | None:
        """Normalize record_to parameter to Path object if provided."""
        return Path(record_to) if record_to else None

    async def _handle_tool_calls(self, tool_calls: "Sequence[ToolCall] | None", includes: Sequence[AgentChunkType], context: "Any | None" = None) -> AsyncGenerator[AgentChunk, None]:  # noqa: ANN401
        """Handle tool calls and yield appropriate chunks."""
        if not tool_calls:
            return

        # Check for transfer_to_agent calls first
        transfer_calls = [tc for tc in tool_calls if tc.function.name == "transfer_to_agent"]
        if transfer_calls:
            # Handle all transfer calls but only execute the first one
            for i, tool_call in enumerate(transfer_calls):
                if i == 0:
                    # Execute the first transfer
                    await self._handle_agent_transfer(tool_call, includes)
                else:
                    # Add response for additional transfer calls without executing them
                    self._add_tool_call_result(
                        call_id=tool_call.id,
                        output="Transfer already executed by previous call",
                    )
            return  # Stop processing other tool calls after transfer

        return_parent_calls = [tc for tc in tool_calls if tc.function.name == "transfer_to_parent"]
        if return_parent_calls:
            # Handle multiple transfer_to_parent calls (only execute the first one)
            for i, tool_call in enumerate(return_parent_calls):
                if i == 0:
                    # Execute the first transfer
                    await self._handle_parent_transfer(tool_call, includes)
                else:
                    # Add response for additional transfer calls without executing them
                    self._add_tool_call_result(
                        call_id=tool_call.id,
                        output="Transfer already executed by previous call",
                    )
            return  # Stop processing other tool calls after transfer

        async for tool_call_chunk in self.agent.handle_tool_calls(tool_calls, context=context):
            # if tool_call_chunk.type == "function_call" and tool_call_chunk.type in includes:
            #     yield tool_call_chunk
            if tool_call_chunk.type == "function_call_output":
                if tool_call_chunk.type in includes:
                    yield tool_call_chunk
                # Add tool result to the last assistant message
                if self.messages and isinstance(self.messages[-1], NewAssistantMessage):
                    tool_result = AssistantToolCallResult(
                        call_id=tool_call_chunk.tool_call_id,
                        output=tool_call_chunk.content,
                        execution_time_ms=tool_call_chunk.execution_time_ms,
                    )
                    self.messages[-1].content.append(tool_result)

    async def _collect_all_chunks(self, stream: AsyncGenerator[AgentChunk, None]) -> list[AgentChunk]:
        """Collect all chunks from an async generator into a list."""
        return [chunk async for chunk in stream]

    def run(
        self,
        user_input: UserInput,
        max_steps: int = 20,
        includes: Sequence[AgentChunkType] | None = None,
        context: "Any | None" = None,  # noqa: ANN401
        record_to: PathLike | str | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        includes = self._normalize_includes(includes)
        if isinstance(user_input, str):
            user_message = NewUserMessage(content=[UserTextContent(text=user_input)])
            self.messages.append(user_message)
        elif isinstance(user_input, (list, tuple)):
            # Handle sequence of messages
            for message in user_input:
                self.append_message(message)
        else:
            # Handle single message (BaseModel, TypedDict, or dict)
            # Type assertion needed due to the complex union type
            self.append_message(user_input)  # type: ignore[arg-type]
        return self._run(max_steps, includes, self._normalize_record_path(record_to), context=context)

    async def _run(self, max_steps: int, includes: Sequence[AgentChunkType], record_to: Path | None = None, context: Any | None = None) -> AsyncGenerator[AgentChunk, None]:  # noqa: ANN401
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        logger.debug(f"Running agent with messages: {self.messages}")
        steps = 0
        finish_reason = None

        # Determine completion condition based on agent configuration
        completion_condition = getattr(self.agent, "completion_condition", "stop")

        def is_finish() -> bool:
            if completion_condition == "call":
                # Check if wait_for_user was called in the last assistant message
                if self.messages and isinstance(self.messages[-1], NewAssistantMessage):
                    for content_item in self.messages[-1].content:
                        if content_item.type == "tool_call_result" and self._get_tool_call_name_by_id(content_item.call_id) == "wait_for_user":
                            return True
                return False
            return finish_reason == "stop"

        while not is_finish() and steps < max_steps:
            logger.debug(f"Step {steps}: finish_reason={finish_reason}, is_finish()={is_finish()}")
            # Convert to legacy format only when needed for LLM communication
            # This allows us to keep the new format internally but ensures compatibility
            match self.api:
                case "completion":
                    resp = await self.agent.completion(self.messages, record_to_file=record_to)
                case "responses":
                    resp = await self.agent.responses(self.messages, record_to_file=record_to)
                case _:
                    msg = f"Unknown API type: {self.api}"
                    raise ValueError(msg)
            async for chunk in resp:
                if chunk.type in includes:
                    yield chunk
                if chunk.type == "assistant_message":
                    # Start or update assistant message in new format
                    meta = AssistantMessageMeta(
                        sent_at=chunk.message.meta.sent_at,
                        latency_ms=getattr(chunk.message.meta, "latency_ms", None),
                        total_time_ms=getattr(chunk.message.meta, "output_time_ms", None),
                    )
                    # If we already have a current assistant message, just update its metadata
                    if self._current_assistant_message is not None:
                        self._current_assistant_message.meta = meta
                    else:
                        # Extract text content from the new message format
                        text_content = ""
                        if chunk.message.content:
                            for item in chunk.message.content:
                                if hasattr(item, "type") and item.type == "text":
                                    text_content = item.text
                                    break
                        self._start_assistant_message(text_content, meta)
                if chunk.type == "content_delta":
                    # Accumulate text content to current assistant message
                    self._add_text_content_to_current_assistant_message(chunk.delta)
                if chunk.type == "function_call":
                    # Add tool call to current assistant message
                    # Keep arguments as string for compatibility with funcall library
                    tool_call = AssistantToolCall(
                        call_id=chunk.call_id,
                        name=chunk.name,
                        arguments=chunk.arguments or "{}",
                    )
                    self._add_to_current_assistant_message(tool_call)
                if chunk.type == "usage":
                    # Update the last assistant message with usage data and output_time_ms
                    usage_time = datetime.now(timezone.utc)
                    for i in range(len(self.messages) - 1, -1, -1):
                        current_message = self.messages[i]
                        if isinstance(current_message, NewAssistantMessage):
                            # Update usage information
                            if current_message.meta.usage is None:
                                current_message.meta.usage = MessageUsage()
                            current_message.meta.usage.input_tokens = chunk.usage.input_tokens
                            current_message.meta.usage.output_tokens = chunk.usage.output_tokens
                            current_message.meta.usage.total_tokens = (chunk.usage.input_tokens or 0) + (chunk.usage.output_tokens or 0)

                            # Calculate output_time_ms if latency_ms is available
                            if current_message.meta.latency_ms is not None:
                                # We need to calculate from first output to usage time
                                # We'll calculate: usage_time - (sent_at - latency_ms)
                                # This gives us the time from first output to usage completion
                                # sent_at is when the message was completed, so sent_at - latency_ms approximates first output time
                                first_output_time_approx = current_message.meta.sent_at - timedelta(milliseconds=current_message.meta.latency_ms)
                                output_time_ms = int((usage_time - first_output_time_approx).total_seconds() * 1000)
                                current_message.meta.total_time_ms = max(0, output_time_ms)
                            break

            # Finalize assistant message so it can be found in pending function calls
            self._finalize_assistant_message()

            # Check for pending tool calls after processing current assistant message
            pending_tool_calls = self._find_pending_tool_calls()
            logger.debug(f"Found {len(pending_tool_calls)} pending tool calls")
            if pending_tool_calls:
                # Convert to ToolCall format for existing handler
                tool_calls = self._convert_tool_calls_to_tool_calls(pending_tool_calls)
                require_confirm_tools = await self.agent.list_require_confirm_tools(tool_calls)
                if require_confirm_tools:
                    return
                async for tool_chunk in self._handle_tool_calls(tool_calls, includes, context=context):
                    yield tool_chunk
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"
            steps += 1

    async def has_require_confirm_tools(self):
        pending_tool_calls = self._find_pending_tool_calls()
        if not pending_tool_calls:
            return False
        tool_calls = self._convert_tool_calls_to_tool_calls(pending_tool_calls)
        require_confirm_tools = await self.agent.list_require_confirm_tools(tool_calls)
        return bool(require_confirm_tools)

    async def run_continue_until_complete(
        self,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> list[AgentChunk]:
        resp = self.run_continue_stream(max_steps, includes, record_to=record_to)
        return await self._collect_all_chunks(resp)

    def run_continue_stream(
        self,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
        context: "Any | None" = None,  # noqa: ANN401
    ) -> AsyncGenerator[AgentChunk, None]:
        return self._run_continue_stream(max_steps, includes, record_to=record_to, context=context)

    async def _run_continue_stream(
        self,
        max_steps: int = 20,
        includes: Sequence[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
        context: "Any | None" = None,  # noqa: ANN401
    ) -> AsyncGenerator[AgentChunk, None]:
        """Continue running the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        includes = self._normalize_includes(includes)

        # Find pending tool calls in responses format
        pending_tool_calls = self._find_pending_tool_calls()
        if pending_tool_calls:
            # Convert to ToolCall format for existing handler
            tool_calls = self._convert_tool_calls_to_tool_calls(pending_tool_calls)
            async for tool_chunk in self._handle_tool_calls(tool_calls, includes, context=context):
                yield tool_chunk
            async for chunk in self._run(max_steps, includes, self._normalize_record_path(record_to)):
                if chunk.type in includes:
                    yield chunk
        else:
            # Check if there are any messages and what the last message is
            if not self.messages:
                msg = "Cannot continue running without a valid last message from the assistant."
                raise ValueError(msg)

            last_message = self.messages[-1]
            if not (isinstance(last_message, NewAssistantMessage) or (hasattr(last_message, "role") and getattr(last_message, "role", None) == "assistant")):
                msg = "Cannot continue running without a valid last message from the assistant."
                raise ValueError(msg)

            resp = self._run(max_steps=max_steps, includes=includes, record_to=self._normalize_record_path(record_to), context=context)
            async for chunk in resp:
                yield chunk

    async def run_until_complete(
        self,
        user_input: UserInput,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> list[AgentChunk]:
        """Run the agent until it completes and return the final message."""
        resp = self.run(user_input, max_steps, includes, record_to=record_to)
        return await self._collect_all_chunks(resp)

    def _find_pending_tool_calls(self) -> list[AssistantToolCall]:
        """Find tool calls that don't have corresponding results yet."""
        # Find pending calls directly in new format messages
        pending_calls: list[AssistantToolCall] = []

        # Look at the last assistant message for pending tool calls
        if not self.messages:
            return pending_calls

        last_message = self.messages[-1]
        if not isinstance(last_message, NewAssistantMessage):
            return pending_calls

        # Collect tool calls and results from the last assistant message
        tool_calls = {}
        tool_results = set()

        for content_item in last_message.content:
            if content_item.type == "tool_call":
                tool_calls[content_item.call_id] = content_item
            elif content_item.type == "tool_call_result":
                tool_results.add(content_item.call_id)

        # Return tool calls that don't have corresponding results
        return [call for call_id, call in tool_calls.items() if call_id not in tool_results]

    def _get_tool_call_name_by_id(self, call_id: str) -> str | None:
        """Get the tool name for a given call_id from the last assistant message."""
        if not self.messages or not isinstance(self.messages[-1], NewAssistantMessage):
            return None

        for content_item in self.messages[-1].content:
            if content_item.type == "tool_call" and content_item.call_id == call_id:
                return content_item.name
        return None

    def _convert_tool_calls_to_tool_calls(self, tool_calls: list[AssistantToolCall]) -> list[ToolCall]:
        """Convert AssistantToolCall objects to ToolCall objects for compatibility."""

        result_tool_calls = []
        for tc in tool_calls:
            tool_call = ToolCall(
                id=tc.call_id,
                type="function",
                function=ToolCallFunction(
                    name=tc.name,
                    arguments=tc.arguments if isinstance(tc.arguments, str) else str(tc.arguments),
                ),
                index=len(result_tool_calls),
            )
            result_tool_calls.append(tool_call)
        return result_tool_calls

    def set_chat_history(self, messages: Sequence[FlexibleRunnerMessage], root_agent: Agent | None = None) -> None:
        """Set the entire chat history and track the current agent based on function calls.

        This method analyzes the message history to determine which agent should be active
        based on transfer_to_agent and transfer_to_parent function calls.

        Args:
            messages: List of messages to set as the chat history
            root_agent: The root agent to use if no transfers are found. If None, uses self.agent
        """
        # Clear current messages
        self.messages.clear()

        # Set initial agent
        current_agent = root_agent if root_agent is not None else self.agent

        # Add each message and track agent transfers
        for message in messages:
            self.append_message(message)
            current_agent = self._track_agent_transfer_in_message(message, current_agent)

        # Set the current agent based on the tracked transfers
        self.agent = current_agent
        logger.info(f"Chat history set with {len(self.messages)} messages. Current agent: {self.agent.name}")

    def get_messages_dict(self) -> list[dict[str, Any]]:
        """Get the messages in JSONL format."""
        return [msg.model_dump(mode="json") for msg in self.messages]

    def _track_agent_transfer_in_message(self, message: FlexibleRunnerMessage, current_agent: Agent) -> Agent:
        """Track agent transfers in a single message.

        Args:
            message: The message to analyze for transfers
            current_agent: The currently active agent

        Returns:
            The agent that should be active after processing this message
        """
        if isinstance(message, dict):
            return self._track_transfer_from_dict_message(message, current_agent)
        if isinstance(message, NewAssistantMessage):
            return self._track_transfer_from_new_assistant_message(message, current_agent)

        return current_agent

    def _track_transfer_from_new_assistant_message(self, message: NewAssistantMessage, current_agent: Agent) -> Agent:
        """Track transfers from NewAssistantMessage objects."""
        for content_item in message.content:
            if content_item.type == "tool_call":
                if content_item.name == "transfer_to_agent":
                    arguments = content_item.arguments if isinstance(content_item.arguments, str) else str(content_item.arguments)
                    return self._handle_transfer_to_agent_tracking(arguments, current_agent)
                if content_item.name == "transfer_to_parent":
                    return self._handle_transfer_to_parent_tracking(current_agent)
        return current_agent

    def _track_transfer_from_dict_message(self, message: dict[str, Any] | MessageDict, current_agent: Agent) -> Agent:
        """Track transfers from dictionary-format messages."""
        message_type = message.get("type")
        if message_type != "function_call":
            return current_agent

        function_name = message.get("name", "")
        if function_name == "transfer_to_agent":
            return self._handle_transfer_to_agent_tracking(message.get("arguments", ""), current_agent)

        if function_name == "transfer_to_parent":
            return self._handle_transfer_to_parent_tracking(current_agent)

        return current_agent

    def _handle_transfer_to_agent_tracking(self, arguments: str | dict, current_agent: Agent) -> Agent:
        """Handle transfer_to_agent function call tracking."""
        try:
            args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments

            target_agent_name = args_dict.get("name")
            if target_agent_name:
                target_agent = self._find_agent_by_name(current_agent, target_agent_name)
                if target_agent:
                    logger.debug(f"History tracking: Transferring from {current_agent.name} to {target_agent_name}")
                    return target_agent

                logger.warning(f"Target agent '{target_agent_name}' not found in handoffs during history setup")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse transfer_to_agent arguments during history setup: {e}")

        return current_agent

    def _handle_transfer_to_parent_tracking(self, current_agent: Agent) -> Agent:
        """Handle transfer_to_parent function call tracking."""
        if current_agent.parent:
            logger.debug(f"History tracking: Transferring from {current_agent.name} back to parent {current_agent.parent.name}")
            return current_agent.parent

        logger.warning(f"Agent {current_agent.name} has no parent to transfer back to during history setup")
        return current_agent

    def _find_agent_by_name(self, root_agent: Agent, target_name: str) -> Agent | None:
        """Find an agent by name in the handoffs tree starting from root_agent.

        Args:
            root_agent: The root agent to start searching from
            target_name: The name of the agent to find

        Returns:
            The agent if found, None otherwise
        """
        # Check direct handoffs from current agent
        if root_agent.handoffs:
            for agent in root_agent.handoffs:
                if agent.name == target_name:
                    return agent

        # If not found in direct handoffs, check if we need to look in parent's handoffs
        # This handles cases where agents can transfer to siblings
        current = root_agent
        while current.parent is not None:
            current = current.parent
            if current.handoffs:
                for agent in current.handoffs:
                    if agent.name == target_name:
                        return agent

        return None

    def append_message(self, message: FlexibleRunnerMessage) -> None:
        if isinstance(message, NewMessage):
            # Already in new format
            self.messages.append(message)
        elif isinstance(message, dict):
            # Handle different message types from dict
            message_type = message.get("type")
            role = message.get("role")

            if role == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    user_message = NewUserMessage(content=[UserTextContent(text=content)])
                elif isinstance(content, list):
                    # Handle complex content array
                    user_content_items: list[UserMessageContent] = []
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get("type")
                            if item_type in {"input_text", "text"}:
                                user_content_items.append(UserTextContent(text=item.get("text", "")))
                            elif item_type in {"input_image", "image_url"}:
                                if item_type == "image_url":
                                    # Handle completion API format
                                    image_url = item.get("image_url", {})
                                    url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                                    user_content_items.append(UserImageContent(image_url=url))
                                else:
                                    # Handle response API format
                                    user_content_items.append(
                                        UserImageContent(
                                            image_url=item.get("image_url"),
                                            file_id=item.get("file_id"),
                                            detail=item.get("detail", "auto"),
                                        ),
                                    )
                        elif hasattr(item, "type"):
                            # Handle Pydantic models
                            if item.type == "input_text":
                                user_content_items.append(UserTextContent(text=item.text))
                            elif item.type == "input_image":
                                user_content_items.append(
                                    UserImageContent(
                                        image_url=getattr(item, "image_url", None),
                                        file_id=getattr(item, "file_id", None),
                                        detail=getattr(item, "detail", "auto"),
                                    ),
                                )
                        else:
                            # Fallback: convert to text
                            user_content_items.append(UserTextContent(text=str(item)))

                    user_message = NewUserMessage(content=user_content_items)
                else:
                    # Handle non-string, non-list content
                    user_message = NewUserMessage(content=[UserTextContent(text=str(content))])
                self.messages.append(user_message)
            elif role == "system":
                content = message.get("content", "")
                system_message = NewSystemMessage(content=str(content))
                self.messages.append(system_message)
            elif role == "assistant":
                content = message.get("content", "")
                assistant_content_items: list[AssistantMessageContent] = [AssistantTextContent(text=str(content))] if content else []

                # Handle tool calls if present
                if "tool_calls" in message:
                    for tool_call in message.get("tool_calls", []):
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except (json.JSONDecodeError, TypeError):
                            arguments = tool_call["function"]["arguments"]

                        assistant_content_items.append(
                            AssistantToolCall(
                                call_id=tool_call["id"],
                                name=tool_call["function"]["name"],
                                arguments=arguments,
                            ),
                        )

                assistant_message = NewAssistantMessage(content=assistant_content_items)
                self.messages.append(assistant_message)
            elif message_type == "function_call":
                # Handle function_call directly like AgentFunctionToolCallMessage
                # Type guard: ensure we have the right message type
                if "call_id" in message and "name" in message and "arguments" in message:
                    function_call_msg = message  # Type should be FunctionCallDict now
                    if self.messages and isinstance(self.messages[-1], NewAssistantMessage):
                        tool_call = AssistantToolCall(
                            call_id=function_call_msg["call_id"],  # type: ignore
                            name=function_call_msg["name"],  # type: ignore
                            arguments=function_call_msg["arguments"],  # type: ignore
                        )
                        self.messages[-1].content.append(tool_call)
                    else:
                        assistant_message = NewAssistantMessage(
                            content=[
                                AssistantToolCall(
                                    call_id=function_call_msg["call_id"],  # type: ignore
                                    name=function_call_msg["name"],  # type: ignore
                                    arguments=function_call_msg["arguments"],  # type: ignore
                                ),
                            ],
                        )
                        self.messages.append(assistant_message)
            elif message_type == "function_call_output":
                # Handle function_call_output directly like AgentFunctionCallOutput
                # Type guard: ensure we have the right message type
                if "call_id" in message and "output" in message:
                    function_output_msg = message  # Type should be FunctionCallOutputDict now
                    if self.messages and isinstance(self.messages[-1], NewAssistantMessage):
                        tool_result = AssistantToolCallResult(
                            call_id=function_output_msg["call_id"],  # type: ignore
                            output=function_output_msg["output"],  # type: ignore
                        )
                        self.messages[-1].content.append(tool_result)
                    else:
                        assistant_message = NewAssistantMessage(
                            content=[
                                AssistantToolCallResult(
                                    call_id=function_output_msg["call_id"],  # type: ignore
                                    output=function_output_msg["output"],  # type: ignore
                                ),
                            ],
                        )
                        self.messages.append(assistant_message)
            else:
                msg = "Message must have a 'role' or 'type' field."
                raise ValueError(msg)
        else:
            msg = f"Unsupported message type: {type(message)}"
            raise TypeError(msg)

    async def _handle_agent_transfer(self, tool_call: ToolCall, _includes: Sequence[AgentChunkType]) -> None:
        """Handle agent transfer when transfer_to_agent tool is called.

        Args:
            tool_call: The transfer_to_agent tool call
            _includes: The types of chunks to include in output (unused)
        """

        # Parse the arguments to get the target agent name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
            target_agent_name = arguments.get("name")
        except (json.JSONDecodeError, KeyError):
            logger.error("Failed to parse transfer_to_agent arguments: %s", tool_call.function.arguments)
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output="Failed to parse transfer arguments",
            )
            return

        if not target_agent_name:
            logger.error("No target agent name provided in transfer_to_agent call")
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output="No target agent name provided",
            )
            return

        # Find the target agent in handoffs
        if not self.agent.handoffs:
            logger.error("Current agent has no handoffs configured")
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output="Current agent has no handoffs configured",
            )
            return

        target_agent = None
        for agent in self.agent.handoffs:
            if agent.name == target_agent_name:
                target_agent = agent
                break

        if not target_agent:
            logger.error("Target agent '%s' not found in handoffs", target_agent_name)
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=f"Target agent '{target_agent_name}' not found in handoffs",
            )
            return

        # Execute the transfer tool call to get the result
        try:
            result = await self.agent.fc.call_function_async(
                tool_call.function.name,
                tool_call.function.arguments or "",
            )

            # Add the tool call result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=str(result),
            )

            # Switch to the target agent
            logger.info("Transferring conversation from %s to %s", self.agent.name, target_agent_name)
            self.agent = target_agent

        except Exception as e:
            logger.exception("Failed to execute transfer_to_agent tool call")
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=f"Transfer failed: {e!s}",
            )

    async def _handle_parent_transfer(self, tool_call: ToolCall, _includes: Sequence[AgentChunkType]) -> None:
        """Handle parent transfer when transfer_to_parent tool is called.

        Args:
            tool_call: The transfer_to_parent tool call
            _includes: The types of chunks to include in output (unused)
        """

        # Check if current agent has a parent
        if not self.agent.parent:
            logger.error("Current agent has no parent to transfer back to.")
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output="Current agent has no parent to transfer back to",
            )
            return

        # Execute the transfer tool call to get the result
        try:
            result = await self.agent.fc.call_function_async(
                tool_call.function.name,
                tool_call.function.arguments or "",
            )

            # Add the tool call result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=str(result),
            )

            # Switch to the parent agent
            logger.info("Transferring conversation from %s back to parent %s", self.agent.name, self.agent.parent.name)
            self.agent = self.agent.parent

        except Exception as e:
            logger.exception("Failed to execute transfer_to_parent tool call")
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=f"Transfer to parent failed: {e!s}",
            )
