import json
from collections.abc import AsyncGenerator, Sequence
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

from lite_agent.agent import Agent
from lite_agent.constants import CompletionMode, StreamIncludes, ToolName
from lite_agent.loggers import logger
from lite_agent.types import (
    AgentChunk,
    AgentChunkType,
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    FlexibleInputMessage,
    FlexibleRunnerMessage,
    MessageUsage,
    NewAssistantMessage,
    NewMessage,
    NewSystemMessage,
    NewUserMessage,
    ToolCall,
    ToolCallFunction,
    UserInput,
    UserTextContent,
)
from lite_agent.types.events import AssistantMessageEvent, FunctionCallOutputEvent
from lite_agent.utils.message_builder import MessageBuilder
from lite_agent.utils.message_state_manager import MessageStateManager


class Runner:
    def __init__(self, agent: Agent, api: Literal["completion", "responses"] = "responses", *, streaming: bool = True) -> None:
        self.agent = agent
        self.messages: list[FlexibleRunnerMessage] = []
        self.api = api
        self.streaming = streaming
        self._message_state_manager = MessageStateManager()
        self.usage = MessageUsage(input_tokens=0, output_tokens=0, total_tokens=0)

    async def _start_assistant_message(self, content: str = "", meta: AssistantMessageMeta | None = None) -> None:
        """Start a new assistant message."""
        # Create meta with model information if not provided
        if meta is None:
            meta = AssistantMessageMeta()
            if hasattr(self.agent.client, "model"):
                meta.model = self.agent.client.model
        await self._message_state_manager.start_message(content, meta)

    async def _ensure_current_assistant_message(self) -> NewAssistantMessage:
        """Ensure current assistant message exists and return it."""
        return await self._message_state_manager.ensure_message_exists()

    async def _add_to_current_assistant_message(self, content_item: AssistantTextContent | AssistantToolCall | AssistantToolCallResult) -> None:
        """Add content to the current assistant message."""
        if isinstance(content_item, AssistantTextContent):
            await self._message_state_manager.add_text_delta(content_item.text)
        elif isinstance(content_item, AssistantToolCall):
            await self._message_state_manager.add_tool_call(content_item)
        elif isinstance(content_item, AssistantToolCallResult):
            await self._message_state_manager.add_tool_result(content_item)

    async def _add_text_content_to_current_assistant_message(self, delta: str) -> None:
        """Add text delta to the current assistant message's text content."""
        await self._message_state_manager.add_text_delta(delta)

    async def _finalize_assistant_message(self) -> None:
        """Finalize the current assistant message and add it to messages."""
        finalized_message = await self._message_state_manager.finalize_message()
        if finalized_message is not None:
            self.messages.append(finalized_message)

    def _add_tool_call_result(self, call_id: str, output: str, execution_time_ms: int | None = None) -> None:
        """Add a tool call result to the last assistant message, or create a new one if needed."""
        result = AssistantToolCallResult(
            call_id=call_id,
            output=output,
            execution_time_ms=execution_time_ms,
        )

        if self.messages and isinstance(self.messages[-1], NewAssistantMessage):
            # Add to existing assistant message
            last_message = cast("NewAssistantMessage", self.messages[-1])
            last_message.content.append(result)
            # Ensure model information is set if not already present
            if last_message.meta.model is None and hasattr(self.agent.client, "model"):
                last_message.meta.model = self.agent.client.model
        else:
            # Create new assistant message with just the tool result
            # Include model information if available
            meta = AssistantMessageMeta()
            if hasattr(self.agent.client, "model"):
                meta.model = self.agent.client.model
            assistant_message = NewAssistantMessage(content=[result], meta=meta)
            self.messages.append(assistant_message)

        # For completion API compatibility, create a separate assistant message
        # Note: In the new architecture, we store everything as NewMessage format
        # The conversion to completion format happens when sending to LLM

    def _normalize_includes(self, includes: Sequence[AgentChunkType] | None) -> Sequence[AgentChunkType]:
        """Normalize includes parameter to default if None."""
        return includes if includes is not None else StreamIncludes.DEFAULT_INCLUDES

    def _normalize_record_path(self, record_to: PathLike | str | None) -> Path | None:
        """Normalize record_to parameter to Path object if provided."""
        return Path(record_to) if record_to else None

    async def _handle_tool_calls(self, tool_calls: "Sequence[ToolCall] | None", includes: Sequence[AgentChunkType], context: "Any | None" = None) -> AsyncGenerator[AgentChunk, None]:  # noqa: ANN401
        """Handle tool calls and yield appropriate chunks."""
        if not tool_calls:
            return

        # Check for transfer_to_agent calls first
        transfer_calls = [tc for tc in tool_calls if tc.function.name == ToolName.TRANSFER_TO_AGENT]
        if transfer_calls:
            logger.info(f"Processing {len(transfer_calls)} transfer_to_agent calls")
            # Handle all transfer calls but only execute the first one
            for i, tool_call in enumerate(transfer_calls):
                if i == 0:
                    # Execute the first transfer
                    logger.info(f"Executing agent transfer: {tool_call.function.arguments}")
                    call_id, output = await self._handle_agent_transfer(tool_call)
                    # Generate function_call_output event if in includes
                    if "function_call_output" in includes:
                        yield FunctionCallOutputEvent(
                            tool_call_id=call_id,
                            name=tool_call.function.name,
                            content=output,
                            execution_time_ms=0,  # Transfer operations are typically fast
                        )
                else:
                    # Add response for additional transfer calls without executing them
                    output = "Transfer already executed by previous call"
                    self._add_tool_call_result(
                        call_id=tool_call.id,
                        output=output,
                    )
                    # Generate function_call_output event if in includes
                    if "function_call_output" in includes:
                        yield FunctionCallOutputEvent(
                            tool_call_id=tool_call.id,
                            name=tool_call.function.name,
                            content=output,
                            execution_time_ms=0,
                        )
            return  # Stop processing other tool calls after transfer

        return_parent_calls = [tc for tc in tool_calls if tc.function.name == ToolName.TRANSFER_TO_PARENT]
        if return_parent_calls:
            # Handle multiple transfer_to_parent calls (only execute the first one)
            for i, tool_call in enumerate(return_parent_calls):
                if i == 0:
                    # Execute the first transfer
                    call_id, output = await self._handle_parent_transfer(tool_call)
                    # Generate function_call_output event if in includes
                    if "function_call_output" in includes:
                        yield FunctionCallOutputEvent(
                            tool_call_id=call_id,
                            name=tool_call.function.name,
                            content=output,
                            execution_time_ms=0,  # Transfer operations are typically fast
                        )
                else:
                    # Add response for additional transfer calls without executing them
                    output = "Transfer already executed by previous call"
                    self._add_tool_call_result(
                        call_id=tool_call.id,
                        output=output,
                    )
                    # Generate function_call_output event if in includes
                    if "function_call_output" in includes:
                        yield FunctionCallOutputEvent(
                            tool_call_id=tool_call.id,
                            name=tool_call.function.name,
                            content=output,
                            execution_time_ms=0,
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
                    last_message = cast("NewAssistantMessage", self.messages[-1])
                    last_message.content.append(tool_result)

    async def _collect_all_chunks(self, stream: AsyncGenerator[AgentChunk, None]) -> list[AgentChunk]:
        """Collect all chunks from an async generator into a list."""
        return [chunk async for chunk in stream]

    def run(
        self,
        user_input: UserInput | None = None,
        max_steps: int = 20,
        includes: Sequence[AgentChunkType] | None = None,
        context: "Any | None" = None,  # noqa: ANN401
        record_to: PathLike | str | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk.

        If user_input is None, the method will continue execution from the current state,
        equivalent to calling the continue methods.
        """
        logger.debug(f"Runner.run called with streaming={self.streaming}, api={self.api}")
        includes = self._normalize_includes(includes)

        # If no user input provided, use continue logic
        if user_input is None:
            logger.debug("No user input provided, using continue logic")
            return self._run_continue_stream(max_steps, includes, self._normalize_record_path(record_to), context)

        # Cancel any pending tool calls before processing new user input
        # and yield cancellation events if they should be included
        cancellation_events = self._cancel_pending_tool_calls()

        # We need to handle this differently since run() is not async
        # Store cancellation events to be yielded by _run
        self._pending_cancellation_events = cancellation_events

        # Process user input
        match user_input:
            case str():
                self.messages.append(NewUserMessage(content=[UserTextContent(text=user_input)]))
            case list() | tuple():
                # Handle sequence of messages
                for message in user_input:
                    self.append_message(message)
            case _:
                # Handle single message (BaseModel, TypedDict, or dict)
                self.append_message(user_input)  # type: ignore[arg-type]
        logger.debug("Messages prepared, calling _run")
        return self._run(max_steps, includes, self._normalize_record_path(record_to), context=context)

    async def _run(
        self,
        max_steps: int,
        includes: Sequence[AgentChunkType],
        record_to: Path | None = None,
        context: Any | None = None,  # noqa: ANN401
    ) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        logger.debug(f"Running agent with messages: {self.messages}")

        # First, yield any pending cancellation events
        if hasattr(self, "_pending_cancellation_events"):
            for cancellation_event in self._pending_cancellation_events:
                if "function_call_output" in includes:
                    yield cancellation_event
            # Clear the pending events after yielding
            delattr(self, "_pending_cancellation_events")

        steps = 0
        finish_reason = None

        # Determine completion condition based on agent configuration
        completion_condition = getattr(self.agent, "completion_condition", CompletionMode.STOP)

        def is_finish() -> bool:
            if completion_condition == CompletionMode.CALL:
                # Check if wait_for_user was called in the last assistant message
                if self.messages and isinstance(self.messages[-1], NewAssistantMessage):
                    last_message = self.messages[-1]
                    for content_item in last_message.content:
                        if isinstance(content_item, AssistantToolCallResult) and self._get_tool_call_name_by_id(content_item.call_id) == ToolName.WAIT_FOR_USER:
                            return True
                return False
            return finish_reason == CompletionMode.STOP

        while not is_finish() and steps < max_steps:
            logger.debug(f"Step {steps}: finish_reason={finish_reason}, is_finish()={is_finish()}")
            logger.info(f"Making LLM request: API={self.api}, streaming={self.streaming}, messages={len(self.messages)}")
            match self.api:
                case "completion":
                    logger.debug("Calling agent.completion")
                    resp = await self.agent.completion(
                        self.messages,
                        record_to_file=record_to,
                        streaming=self.streaming,
                    )
                case "responses":
                    logger.debug("Calling agent.responses")
                    resp = await self.agent.responses(
                        self.messages,
                        record_to_file=record_to,
                        streaming=self.streaming,
                    )
                case _:
                    msg = f"Unknown API type: {self.api}"
                    raise ValueError(msg)
            logger.debug("Received response stream from agent, processing chunks...")
            async for chunk in resp:
                # Only log important chunk types to reduce noise
                if chunk.type not in ["response_raw", "content_delta"]:
                    logger.debug(f"Processing chunk: {chunk.type}")
                match chunk.type:
                    case "assistant_message":
                        logger.debug(f"Assistant message chunk: {len(chunk.message.content) if chunk.message.content else 0} content items")
                        # Start or update assistant message in new format
                        # If we already have a current assistant message, just update its metadata
                        current_message = await self._message_state_manager.get_current_message()
                        if current_message is not None:
                            # Preserve all existing metadata and only update specific fields
                            meta_updates = {"sent_at": chunk.message.meta.sent_at}
                            # Only include fields of type datetime in meta_updates
                            # Update int fields separately after update_meta
                            # Preserve other metadata fields like model, usage, etc.
                            for attr in ["model", "usage", "input_tokens", "output_tokens"]:
                                if hasattr(chunk.message.meta, attr):
                                    meta_updates[attr] = getattr(chunk.message.meta, attr)
                            await self._message_state_manager.update_meta(**meta_updates)
                            # Now update int fields directly if present
                            if hasattr(chunk.message.meta, "latency_ms"):
                                await self._message_state_manager.update_meta(latency_ms=chunk.message.meta.latency_ms)
                            if hasattr(chunk.message.meta, "output_time_ms"):
                                await self._message_state_manager.update_meta(total_time_ms=chunk.message.meta.output_time_ms)
                        else:
                            # For non-streaming mode, start with complete message
                            await self._start_assistant_message(meta=chunk.message.meta)
                            # Add all content from the chunk message
                            for content_item in chunk.message.content:
                                await self._add_to_current_assistant_message(content_item)

                        # If model is None, try to get it from agent client
                        current_message = await self._message_state_manager.get_current_message()
                        if current_message is not None and current_message.meta.model is None and hasattr(self.agent.client, "model"):
                            await self._message_state_manager.update_meta(model=self.agent.client.model)
                        # Only yield assistant_message chunk if it's in includes and has content
                        if chunk.type in includes and current_message is not None:
                            # Create a new chunk with the current assistant message content
                            updated_chunk = AssistantMessageEvent(
                                message=current_message,
                            )
                            yield updated_chunk
                    case "content_delta":
                        # Accumulate text content to current assistant message
                        await self._add_text_content_to_current_assistant_message(chunk.delta)
                        # Always yield content_delta chunk if it's in includes
                        if chunk.type in includes:
                            yield chunk
                    case "function_call":
                        logger.debug(f"Function call: {chunk.name}({chunk.arguments or '{}'})")
                        # Add tool call to current assistant message
                        tool_call = AssistantToolCall(
                            call_id=chunk.call_id,
                            name=chunk.name,
                            arguments=chunk.arguments or "{}",
                        )
                        await self._add_to_current_assistant_message(tool_call)
                        # Always yield function_call chunk if it's in includes
                        if chunk.type in includes:
                            yield chunk
                    case "usage":
                        logger.debug(f"Usage: {chunk.usage.input_tokens} input, {chunk.usage.output_tokens} output tokens")
                        # Update the current or last assistant message with usage data and output_time_ms
                        usage_time = datetime.now(timezone.utc)

                        # Always accumulate usage in runner first
                        self.usage.input_tokens = (self.usage.input_tokens or 0) + (chunk.usage.input_tokens or 0)
                        self.usage.output_tokens = (self.usage.output_tokens or 0) + (chunk.usage.output_tokens or 0)
                        self.usage.total_tokens = (self.usage.total_tokens or 0) + (chunk.usage.input_tokens or 0) + (chunk.usage.output_tokens or 0)

                        # Try to find the assistant message to update
                        target_message = None

                        # First check if we have a current assistant message
                        target_message = await self._message_state_manager.get_current_message()
                        if target_message is None:
                            # Otherwise, look for the last assistant message in the list
                            for i in range(len(self.messages) - 1, -1, -1):
                                current_message = self.messages[i]
                                if isinstance(current_message, NewAssistantMessage):
                                    target_message = current_message
                                    break

                        # Update the target message with usage information
                        if target_message is not None:
                            if target_message.meta.usage is None:
                                target_message.meta.usage = MessageUsage()
                            target_message.meta.usage.input_tokens = chunk.usage.input_tokens
                            target_message.meta.usage.output_tokens = chunk.usage.output_tokens
                            target_message.meta.usage.total_tokens = (chunk.usage.input_tokens or 0) + (chunk.usage.output_tokens or 0)

                            # Calculate output_time_ms if latency_ms is available
                            if target_message.meta.latency_ms is not None:
                                # We need to calculate from first output to usage time
                                # We'll calculate: usage_time - (sent_at - latency_ms)
                                # This gives us the time from first output to usage completion
                                # sent_at is when the message was completed, so sent_at - latency_ms approximates first output time
                                first_output_time_approx = target_message.meta.sent_at - timedelta(milliseconds=target_message.meta.latency_ms)
                                output_time_ms = int((usage_time - first_output_time_approx).total_seconds() * 1000)
                                target_message.meta.total_time_ms = max(0, output_time_ms)
                        # Always yield usage chunk if it's in includes
                        if chunk.type in includes:
                            yield chunk
                    case "timing":
                        # Update timing information in current assistant message
                        current_message = await self._message_state_manager.get_current_message()
                        if current_message is not None:
                            await self._message_state_manager.update_meta(latency_ms=chunk.timing.latency_ms, total_time_ms=chunk.timing.output_time_ms)
                        # Also try to update the last assistant message if no current message
                        elif self.messages and isinstance(self.messages[-1], NewAssistantMessage):
                            last_message = cast("NewAssistantMessage", self.messages[-1])
                            last_message.meta.latency_ms = chunk.timing.latency_ms
                            last_message.meta.total_time_ms = chunk.timing.output_time_ms
                        # Always yield timing chunk if it's in includes
                        if chunk.type in includes:
                            yield chunk
                    case _ if chunk.type in includes:
                        yield chunk

            # Finalize assistant message so it can be found in pending function calls
            await self._finalize_assistant_message()

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
                finish_reason = CompletionMode.STOP
            steps += 1

    async def has_require_confirm_tools(self):
        pending_tool_calls = self._find_pending_tool_calls()
        if not pending_tool_calls:
            return False
        tool_calls = self._convert_tool_calls_to_tool_calls(pending_tool_calls)
        require_confirm_tools = await self.agent.list_require_confirm_tools(tool_calls)
        return bool(require_confirm_tools)

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

            resp = self._run(max_steps=max_steps, includes=includes, record_to=self._normalize_record_path(record_to), context=context)
            async for chunk in resp:
                yield chunk

    async def run_until_complete(
        self,
        user_input: UserInput | None = None,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> list[AgentChunk]:
        """Run the agent until it completes and return the final message."""
        resp = self.run(user_input, max_steps, includes, record_to=record_to)
        return await self._collect_all_chunks(resp)

    def _analyze_last_assistant_message(self) -> tuple[list[AssistantToolCall], dict[str, str]]:
        """Analyze the last assistant message and return pending tool calls and tool call map."""
        if not self.messages or not isinstance(self.messages[-1], NewAssistantMessage):
            return [], {}

        tool_calls = {}
        tool_results = set()
        tool_call_names = {}

        last_message = self.messages[-1]
        for content_item in last_message.content:
            if isinstance(content_item, AssistantToolCall):
                tool_calls[content_item.call_id] = content_item
                tool_call_names[content_item.call_id] = content_item.name
            elif isinstance(content_item, AssistantToolCallResult):
                tool_results.add(content_item.call_id)

        # Return pending tool calls and tool call names map
        pending_calls = [call for call_id, call in tool_calls.items() if call_id not in tool_results]
        return pending_calls, tool_call_names

    def _find_pending_tool_calls(self) -> list[AssistantToolCall]:
        """Find tool calls that don't have corresponding results yet."""
        pending_calls, _ = self._analyze_last_assistant_message()
        return pending_calls

    def _get_tool_call_name_by_id(self, call_id: str) -> str | None:
        """Get the tool name for a given call_id from the last assistant message."""
        _, tool_call_names = self._analyze_last_assistant_message()
        return tool_call_names.get(call_id)

    def _cancel_pending_tool_calls(self) -> list[FunctionCallOutputEvent]:
        """Cancel all pending tool calls by adding cancellation results.

        Returns:
            List of FunctionCallOutputEvent for each cancelled tool call
        """
        pending_tool_calls = self._find_pending_tool_calls()
        if not pending_tool_calls:
            return []

        logger.debug(f"Cancelling {len(pending_tool_calls)} pending tool calls due to new user input")

        cancellation_events = []
        for tool_call in pending_tool_calls:
            output = "Operation cancelled by user - new input provided"
            self._add_tool_call_result(
                call_id=tool_call.call_id,
                output=output,
                execution_time_ms=0,
            )

            # Create cancellation event
            cancellation_event = FunctionCallOutputEvent(
                tool_call_id=tool_call.call_id,
                name=tool_call.name,
                content=output,
                execution_time_ms=0,
            )
            cancellation_events.append(cancellation_event)

        return cancellation_events

    def _convert_tool_calls_to_tool_calls(self, tool_calls: list[AssistantToolCall]) -> list[ToolCall]:
        """Convert AssistantToolCall objects to ToolCall objects for compatibility."""
        return [
            ToolCall(
                id=tc.call_id,
                type="function",
                function=ToolCallFunction(
                    name=tc.name,
                    arguments=tc.arguments if isinstance(tc.arguments, str) else str(tc.arguments),
                ),
                index=i,
            )
            for i, tc in enumerate(tool_calls)
        ]

    def set_chat_history(self, messages: Sequence[FlexibleInputMessage], root_agent: Agent | None = None) -> None:
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
        for input_message in messages:
            # Store length before adding to get the added message
            prev_length = len(self.messages)
            self.append_message(input_message)

            # Track transfers using the converted message (now in self.messages)
            if len(self.messages) > prev_length:
                converted_message = self.messages[-1]  # Get the last added message
                current_agent = self._track_agent_transfer_in_message(converted_message, current_agent)

        # Set the current agent based on the tracked transfers
        self.agent = current_agent
        logger.info(f"Chat history set with {len(self.messages)} messages. Current agent: {self.agent.name}")

    def get_messages(self) -> list[NewMessage]:
        """Get the messages as NewMessage objects.

        Only returns NewMessage objects, filtering out any dict or other legacy formats.
        """
        return [msg for msg in self.messages if isinstance(msg, NewMessage)]

    def get_dict_messages(self) -> list[dict[str, Any]]:
        """Get the messages in JSONL format."""
        result = []
        for msg in self.messages:
            if hasattr(msg, "model_dump"):
                result.append(msg.model_dump(mode="json"))
            elif isinstance(msg, dict):
                result.append(msg)
            else:
                # Fallback for any other message types
                result.append(dict(msg))
        return result

    def add_user_message(self, text: str) -> None:
        """Convenience method to add a user text message."""
        message = NewUserMessage(content=[UserTextContent(text=text)])
        self.append_message(message)

    def add_assistant_message(self, text: str) -> None:
        """Convenience method to add an assistant text message."""
        message = NewAssistantMessage(content=[AssistantTextContent(text=text)])
        self.append_message(message)

    def add_system_message(self, content: str) -> None:
        """Convenience method to add a system message."""
        message = NewSystemMessage(content=content)
        self.append_message(message)

    def _track_agent_transfer_in_message(self, message: FlexibleRunnerMessage, current_agent: Agent) -> Agent:
        """Track agent transfers in a single message.

        Args:
            message: The message to analyze for transfers
            current_agent: The currently active agent

        Returns:
            The agent that should be active after processing this message
        """
        if isinstance(message, NewAssistantMessage):
            return self._track_transfer_from_new_assistant_message(message, current_agent)

        return current_agent

    def _track_transfer_from_new_assistant_message(self, message: NewAssistantMessage, current_agent: Agent) -> Agent:
        """Track transfers from NewAssistantMessage objects."""
        for content_item in message.content:
            if content_item.type == "tool_call":
                if content_item.name == ToolName.TRANSFER_TO_AGENT:
                    arguments = content_item.arguments if isinstance(content_item.arguments, str) else str(content_item.arguments)
                    return self._handle_transfer_to_agent_tracking(arguments, current_agent)
                if content_item.name == ToolName.TRANSFER_TO_PARENT:
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

    def append_message(self, message: FlexibleInputMessage) -> None:
        """Append a message to the conversation history.

        Accepts both NewMessage format and dict format (which will be converted internally).
        """
        if isinstance(message, NewMessage):
            self.messages.append(message)
        elif isinstance(message, dict):
            # Convert dict to NewMessage using MessageBuilder
            role = message.get("role", "").lower()
            if role == "user":
                converted_message = MessageBuilder.build_user_message_from_dict(message)
            elif role == "assistant":
                converted_message = MessageBuilder.build_assistant_message_from_dict(message)
            elif role == "system":
                converted_message = MessageBuilder.build_system_message_from_dict(message)
            else:
                msg = f"Unsupported message role: {role}. Must be 'user', 'assistant', or 'system'."
                raise ValueError(msg)

            self.messages.append(converted_message)
        else:
            msg = f"Unsupported message type: {type(message)}. Supports NewMessage types and dict."
            raise TypeError(msg)

    async def _handle_agent_transfer(self, tool_call: ToolCall) -> tuple[str, str]:
        """Handle agent transfer when transfer_to_agent tool is called.

        Args:
            tool_call: The transfer_to_agent tool call

        Returns:
            Tuple of (call_id, output) for the tool call result
        """

        # Parse the arguments to get the target agent name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
            target_agent_name = arguments.get("name")
        except (json.JSONDecodeError, KeyError):
            logger.error("Failed to parse transfer_to_agent arguments: %s", tool_call.function.arguments)
            output = "Failed to parse transfer arguments"
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )
            return tool_call.id, output

        if not target_agent_name:
            logger.error("No target agent name provided in transfer_to_agent call")
            output = "No target agent name provided"
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )
            return tool_call.id, output

        # Find the target agent in handoffs
        if not self.agent.handoffs:
            logger.error("Current agent has no handoffs configured")
            output = "Current agent has no handoffs configured"
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )
            return tool_call.id, output

        target_agent = None
        for agent in self.agent.handoffs:
            if agent.name == target_agent_name:
                target_agent = agent
                break

        if not target_agent:
            logger.error("Target agent '%s' not found in handoffs", target_agent_name)
            output = f"Target agent '{target_agent_name}' not found in handoffs"
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )
            return tool_call.id, output

        # Execute the transfer tool call to get the result
        try:
            result = await self.agent.fc.call_function_async(
                tool_call.function.name,
                tool_call.function.arguments or "",
            )

            output = str(result)
            # Add the tool call result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )

            # Switch to the target agent
            logger.info("Transferring conversation from %s to %s", self.agent.name, target_agent_name)
            self.agent = target_agent

        except Exception as e:
            logger.exception("Failed to execute transfer_to_agent tool call")
            output = f"Transfer failed: {e!s}"
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )
            return tool_call.id, output
        else:
            return tool_call.id, output

    async def _handle_parent_transfer(self, tool_call: ToolCall) -> tuple[str, str]:
        """Handle parent transfer when transfer_to_parent tool is called.

        Args:
            tool_call: The transfer_to_parent tool call

        Returns:
            Tuple of (call_id, output) for the tool call result
        """

        # Check if current agent has a parent
        if not self.agent.parent:
            logger.error("Current agent has no parent to transfer back to.")
            output = "Current agent has no parent to transfer back to"
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )
            return tool_call.id, output

        # Execute the transfer tool call to get the result
        try:
            result = await self.agent.fc.call_function_async(
                tool_call.function.name,
                tool_call.function.arguments or "",
            )

            output = str(result)
            # Add the tool call result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )

            # Switch to the parent agent
            logger.info("Transferring conversation from %s back to parent %s", self.agent.name, self.agent.parent.name)
            self.agent = self.agent.parent

        except Exception as e:
            logger.exception("Failed to execute transfer_to_parent tool call")
            output = f"Transfer to parent failed: {e!s}"
            # Add error result to messages
            self._add_tool_call_result(
                call_id=tool_call.id,
                output=output,
            )
            return tool_call.id, output
        else:
            return tool_call.id, output
