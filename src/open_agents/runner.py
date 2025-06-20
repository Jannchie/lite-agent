from collections.abc import AsyncGenerator
from typing import Literal

from easy_agent.agent import Agent
from easy_agent.chunk_handler import AgentChunk
from easy_agent.types import AgentToolCallMessage, RunnerMessages


class Runner:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.messages: RunnerMessages = []

    def run_stream(
        self,
        user_input: RunnerMessages | str,
        max_steps: int = 20,
        includes: list[Literal["usage", "final_message", "tool_call", "tool_call_result"]] | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        if includes is None:
            includes = ["final_message", "usage", "tool_call", "tool_call_result", "tool_call_delta", "content_delta"]
        if isinstance(user_input, str):
            self.messages.append({"role": "user", "content": user_input})
        else:
            self.messages = user_input

        return self._run_aiter(max_steps, includes)

    async def _run_aiter(self, max_steps: int, includes: list[Literal["usage", "final_message", "tool_call", "tool_call_result"]]) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        steps = 0
        finish_reason = None
        while finish_reason != "stop" and steps < max_steps:
            resp = await self.agent.stream_async(self.messages)
            async for chunk in resp:
                if chunk["type"] == "final_message":
                    message = chunk["message"]
                    self.messages.append(message.model_dump())
                    finish_reason = chunk["finish_reason"]
                elif chunk["type"] == "tool_call_result":
                    self.messages.append(
                        AgentToolCallMessage(
                            role="tool",
                            tool_call_id=chunk["tool_call_id"],
                            content=chunk["content"],
                        ),
                    )
                if chunk["type"] in includes:
                    yield chunk
            steps += 1
