from collections.abc import AsyncGenerator, Sequence
from os import PathLike
from pathlib import Path

from lite_agent.agent import Agent
from lite_agent.loggers import logger
from lite_agent.types import AgentAssistantMessage, AgentChunk, AgentChunkType, AgentSystemMessage, AgentToolCallMessage, AgentUserMessage, RunnerMessage, RunnerMessages

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

    def run_stream(
        self,
        user_input: RunnerMessages | str,
        max_steps: int = 20,
        includes: Sequence[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        if includes is None:
            includes = DEFAULT_INCLUDES
        if isinstance(user_input, str):
            self.messages.append(AgentUserMessage(role="user", content=user_input))
        else:
            for message in user_input:
                self.append_message(message)
        return self._run_stream(max_steps, includes, record_to=Path(record_to) if record_to else None)

    async def _run_stream(self, max_steps: int, includes: Sequence[AgentChunkType], record_to: Path | None = None) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        logger.debug(f"Running agent with messages: {self.messages}")
        steps = 0
        finish_reason = None

        while finish_reason != "stop" and steps < max_steps:
            resp = await self.agent.stream_async(self.messages, record_to_file=record_to)
            async for chunk in resp:
                if chunk.type in includes:
                    yield chunk

                if chunk.type == "final_message":
                    message = chunk.message
                    self.messages.append(message)  # type: ignore
                    finish_reason = chunk.finish_reason
                    if finish_reason == "tool_calls":
                        # Handle tool calls if the finish reason is tool_calls
                        require_confirm_tools = await self.agent.list_require_confirm_tools(message.tool_calls)
                        if require_confirm_tools:
                            return
                        async for tool_call_chunk in self.agent.handle_tool_calls(message.tool_calls):
                            if tool_call_chunk.type == "tool_call":
                                yield tool_call_chunk
                            if tool_call_chunk.type == "tool_call_result":
                                yield tool_call_chunk
                                self.messages.append(
                                    AgentToolCallMessage(
                                        role="tool",
                                        tool_call_id=tool_call_chunk.tool_call_id,
                                        content=tool_call_chunk.content,
                                    ),
                                )
            steps += 1

    async def run_continue_until_complete(
        self,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> list[AgentChunk]:
        resp = self.run_continue_stream(max_steps, includes, record_to=record_to)
        return [chunk async for chunk in resp]

    def run_continue_stream(
        self,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        return self._run_continue_stream(max_steps, includes, record_to=record_to)

    async def _run_continue_stream(
        self,
        max_steps: int = 20,
        includes: Sequence[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Continue running the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        if includes is None:
            includes = DEFAULT_INCLUDES
        last_message = self.messages[-1] if self.messages else None
        if not last_message or last_message.role != "assistant":
            msg = "Cannot continue running without a valid last message from the assistant."
            raise ValueError(msg)
        async for tool_call_chunk in self.agent.handle_tool_calls(last_message.tool_calls):
            if tool_call_chunk.type == "tool_call":
                yield tool_call_chunk
            if tool_call_chunk.type == "tool_call_result":
                yield tool_call_chunk
                self.messages.append(
                    AgentToolCallMessage(
                        role="tool",
                        tool_call_id=tool_call_chunk.tool_call_id,
                        content=tool_call_chunk.content,
                    ),
                )
        async for chunk in self._run_stream(max_steps, includes, record_to=Path(record_to) if record_to else None):
            if chunk.type in includes:
                yield chunk

    async def run_until_complete(
        self,
        user_input: RunnerMessages | str,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
        record_to: PathLike | str | None = None,
    ) -> list[AgentChunk]:
        """Run the agent until it completes and return the final message."""
        resp = self.run_stream(user_input, max_steps, includes, record_to=record_to)
        return [chunk async for chunk in resp]

    def append_message(self, message: RunnerMessage | dict) -> None:
        if isinstance(message, RunnerMessage):
            self.messages.append(message)
        elif isinstance(message, dict):
            role = message.get("role")
            if not role:
                msg = "Message must have a 'role' field."
                raise ValueError(msg)
            if role == "user":
                self.messages.append(AgentUserMessage.model_validate(message))
            elif role == "assistant":
                self.messages.append(AgentAssistantMessage.model_validate(message))
            elif role == "tool":
                self.messages.append(AgentToolCallMessage.model_validate(message))
            elif role == "system":
                self.messages.append(AgentSystemMessage.model_validate(message))
