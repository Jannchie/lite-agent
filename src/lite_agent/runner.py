from collections.abc import AsyncGenerator

from lite_agent.agent import Agent
from lite_agent.loggers import logger
from lite_agent.types import AgentChunk, AgentChunkType, AgentToolCallMessage, RunnerMessages, ToolCall


class Runner:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.messages: RunnerMessages = []

    def run_stream(
        self,
        user_input: RunnerMessages | str,
        max_steps: int = 20,
        includes: list[AgentChunkType] | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        if includes is None:
            includes = ["final_message", "usage", "tool_call", "tool_call_result", "tool_call_delta", "content_delta", "require_confirm"]
        if isinstance(user_input, str):
            self.messages.append({"role": "user", "content": user_input})
        else:
            self.messages = user_input

        return self._run_aiter(max_steps, includes)

    def run_continue(self, max_steps: int = 20, includes: list[AgentChunkType] | None = None) -> AsyncGenerator[AgentChunk, None]:
        """Continue running the agent with the current messages."""
        return self._run_aiter(max_steps, includes or ["final_message", "usage", "tool_call", "tool_call_result", "tool_call_delta", "content_delta", "require_confirm"])

    async def run_until_complete(self, user_input: RunnerMessages | str, max_steps: int = 20, includes: list[AgentChunkType] | None = None) -> list[AgentChunk]:
        """Run the agent until it completes and return the final message."""
        resp = self.run_stream(user_input, max_steps, includes)
        return [chunk async for chunk in resp]

    async def _run_aiter(self, max_steps: int, includes: list[AgentChunkType]) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        logger.debug(f"Running agent with messages: {self.messages}")
        steps = 0
        finish_reason = None

        while finish_reason != "stop" and steps < max_steps:
            resp = await self.agent.stream_async(self.messages)
            async for chunk in resp:
                if chunk["type"] in includes:
                    yield chunk

                if chunk["type"] == "final_message":
                    message = chunk["message"]
                    self.messages.append(message.model_dump())  # type: ignore
                    finish_reason = chunk["finish_reason"]
                    if finish_reason == "tool_calls":
                        # Handle tool calls if the finish reason is tool_calls
                        require_confirm_tools = await self.agent.list_require_confirm_tools(message.tool_calls)
                        if require_confirm_tools:
                            return
                        async for tool_call_chunk in self._handle_tool_calls(chunk["message"].tool_calls):
                            if tool_call_chunk["type"] == "tool_call":
                                yield tool_call_chunk
                            if tool_call_chunk["type"] == "tool_call_result":
                                yield tool_call_chunk
                                self.messages.append(
                                    AgentToolCallMessage(
                                        role="tool",
                                        tool_call_id=tool_call_chunk["tool_call_id"],
                                        content=tool_call_chunk["content"],
                                    ),
                                )
            steps += 1

    async def _handle_tool_calls(self, tool_calls: list[ToolCall] | None) -> AsyncGenerator[AgentChunk, None]:
        """Handle tool calls and yield tool call chunks and results."""
        async for tool_call_chunk in self.agent.handle_tool_calls(tool_calls):
            if tool_call_chunk["type"] == "tool_call":
                yield tool_call_chunk
            if tool_call_chunk["type"] == "tool_call_result":
                yield tool_call_chunk
                self.messages.append(
                    AgentToolCallMessage(
                        role="tool",
                        tool_call_id=tool_call_chunk["tool_call_id"],
                        content=tool_call_chunk["content"],
                    ),
                )
