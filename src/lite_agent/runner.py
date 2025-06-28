from collections.abc import AsyncGenerator, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lite_agent.agent import Agent
from lite_agent.loggers import logger
from lite_agent.types import (
    AgentAssistantMessage,
    AgentChunk,
    AgentChunkType,
    AgentFunctionCallOutput,
    AgentFunctionToolCallMessage,
    AgentSystemMessage,
    AgentUserMessage,
    RunnerMessage,
    RunnerMessages,
    ToolCall,
    ToolCallFunction,
)

if TYPE_CHECKING:
    from lite_agent.types import AssistantMessage

DEFAULT_INCLUDES: tuple[AgentChunkType, ...] = (
    "completion_raw",
    "usage",
    "final_message",
    "tool_call",
    "tool_call_result",
    "content_delta",
    "tool_call_delta",
)


class Runner:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.messages: list[RunnerMessage] = []

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
        async for tool_call_chunk in self.agent.handle_tool_calls(tool_calls, context=context):
            if tool_call_chunk.type == "tool_call" and tool_call_chunk.type in includes:
                yield tool_call_chunk
            if tool_call_chunk.type == "tool_call_result":
                if tool_call_chunk.type in includes:
                    yield tool_call_chunk
                # Create function call output in responses format
                self.messages.append(
                    AgentFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_call_chunk.tool_call_id,
                        output=tool_call_chunk.content,
                    ),
                )

    async def _collect_all_chunks(self, stream: AsyncGenerator[AgentChunk, None]) -> list[AgentChunk]:
        """Collect all chunks from an async generator into a list."""
        return [chunk async for chunk in stream]

    def run(
        self,
        user_input: RunnerMessages | str,
        max_steps: int = 20,
        includes: Sequence[AgentChunkType] | None = None,
        context: "Any | None" = None,  # noqa: ANN401
        record_to: PathLike | str | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        includes = self._normalize_includes(includes)
        if isinstance(user_input, str):
            self.messages.append(AgentUserMessage(role="user", content=user_input))
        else:
            for message in user_input:
                self.append_message(message)
        return self._run(max_steps, includes, self._normalize_record_path(record_to), context=context)

    async def _run(self, max_steps: int, includes: Sequence[AgentChunkType], record_to: Path | None = None, context: "Any | None" = None) -> AsyncGenerator[AgentChunk, None]:  # noqa: ANN401
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        logger.debug(f"Running agent with messages: {self.messages}")
        steps = 0
        finish_reason = None

        while finish_reason != "stop" and steps < max_steps:
            resp = await self.agent.completion(self.messages, record_to_file=record_to)
            async for chunk in resp:
                if chunk.type in includes:
                    yield chunk

                if chunk.type == "final_message":
                    message = chunk.message
                    # Convert to responses format and add to messages
                    await self._convert_final_message_to_responses_format(message)
                    finish_reason = chunk.finish_reason
                    if finish_reason == "tool_calls":
                        # Find pending function calls in responses format
                        pending_function_calls = self._find_pending_function_calls()
                        if pending_function_calls:
                            # Convert to ToolCall format for existing handler
                            tool_calls = self._convert_function_calls_to_tool_calls(pending_function_calls)
                            require_confirm_tools = await self.agent.list_require_confirm_tools(tool_calls)
                            if require_confirm_tools:
                                return
                            async for tool_chunk in self._handle_tool_calls(tool_calls, includes, context=context):
                                yield tool_chunk
            steps += 1

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

        # Find pending function calls in responses format
        pending_function_calls = self._find_pending_function_calls()
        if pending_function_calls:
            # Convert to ToolCall format for existing handler
            tool_calls = self._convert_function_calls_to_tool_calls(pending_function_calls)
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
            if not (isinstance(last_message, AgentAssistantMessage) or (hasattr(last_message, "role") and getattr(last_message, "role", None) == "assistant")):
                msg = "Cannot continue running without a valid last message from the assistant."
                raise ValueError(msg)

            # If we have an assistant message but no pending function calls,
            # that means there's nothing to continue
            msg = "Cannot continue running without pending function calls."
            raise ValueError(msg)

    async def run_until_complete(
        self,
        user_input: RunnerMessages | str,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> list[AgentChunk]:
        """Run the agent until it completes and return the final message."""
        resp = self.run(user_input, max_steps, includes, record_to=record_to)
        return await self._collect_all_chunks(resp)

    async def _convert_final_message_to_responses_format(self, message: "AssistantMessage") -> None:
        """Convert a completions format final message to responses format messages."""
        # The final message from the stream handler might still contain tool_calls
        # We need to convert it to responses format
        if hasattr(message, "tool_calls") and message.tool_calls:
            # Add the assistant message without tool_calls
            assistant_msg = AgentAssistantMessage(
                role="assistant",
                content=message.content,
            )
            self.messages.append(assistant_msg)

            # Add function call messages
            for tool_call in message.tool_calls:
                function_call_msg = AgentFunctionToolCallMessage(
                    type="function_call",
                    function_call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments or "",
                    content="",
                )
                self.messages.append(function_call_msg)
        else:
            # Regular assistant message without tool calls
            assistant_msg = AgentAssistantMessage(
                role="assistant",
                content=message.content,
            )
            self.messages.append(assistant_msg)

    def _find_pending_function_calls(self) -> list:
        """Find function call messages that don't have corresponding outputs yet."""
        function_calls: list[AgentFunctionToolCallMessage] = []
        function_call_ids = set()

        # Collect all function call messages
        for msg in reversed(self.messages):
            if isinstance(msg, AgentFunctionToolCallMessage):
                function_calls.append(msg)
                function_call_ids.add(msg.function_call_id)
            elif isinstance(msg, AgentFunctionCallOutput):
                # Remove the corresponding function call from our list
                function_call_ids.discard(msg.call_id)
            elif isinstance(msg, AgentAssistantMessage):
                # Stop when we hit the assistant message that initiated these calls
                break

        # Return only function calls that don't have outputs yet
        return [fc for fc in function_calls if fc.function_call_id in function_call_ids]

    def _convert_function_calls_to_tool_calls(self, function_calls: list[AgentFunctionToolCallMessage]) -> list[ToolCall]:
        """Convert function call messages to ToolCall objects for compatibility."""

        tool_calls = []
        for fc in function_calls:
            tool_call = ToolCall(
                id=fc.function_call_id,
                type="function",
                function=ToolCallFunction(
                    name=fc.name,
                    arguments=fc.arguments,
                ),
                index=len(tool_calls),
            )
            tool_calls.append(tool_call)
        return tool_calls

    def append_message(self, message: RunnerMessage | dict) -> None:
        if isinstance(message, RunnerMessage):
            self.messages.append(message)
        elif isinstance(message, dict):
            # Handle different message types
            message_type = message.get("type")
            role = message.get("role")

            if message_type == "function_call":
                # Function call message
                self.messages.append(AgentFunctionToolCallMessage.model_validate(message))
            elif message_type == "function_call_output":
                # Function call output message
                self.messages.append(AgentFunctionCallOutput.model_validate(message))
            elif role == "assistant" and "tool_calls" in message:
                # Legacy assistant message with tool_calls - convert to responses format
                # Add assistant message without tool_calls
                assistant_msg = AgentAssistantMessage(
                    role="assistant",
                    content=message.get("content", ""),
                )
                self.messages.append(assistant_msg)

                # Convert tool_calls to function call messages
                for tool_call in message.get("tool_calls", []):
                    function_call_msg = AgentFunctionToolCallMessage(
                        type="function_call",
                        function_call_id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                        content="",
                    )
                    self.messages.append(function_call_msg)
            elif role:
                # Regular role-based message
                role_to_message_class = {
                    "user": AgentUserMessage,
                    "assistant": AgentAssistantMessage,
                    "system": AgentSystemMessage,
                }

                message_class = role_to_message_class.get(role)
                if message_class:
                    self.messages.append(message_class.model_validate(message))
                else:
                    msg = f"Unsupported message role: {role}"
                    raise ValueError(msg)
            else:
                msg = "Message must have a 'role' or 'type' field."
                raise ValueError(msg)
