from collections.abc import AsyncGenerator

from lite_agent.agent import Agent
from lite_agent.types import AgentChunk, AgentChunkType, AgentToolCallMessage, RunnerMessages


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
            includes = ["final_message", "usage", "tool_call", "tool_call_result", "tool_call_delta", "content_delta"]
        if isinstance(user_input, str):
            self.messages.append({"role": "user", "content": user_input})
        else:
            self.messages = user_input

        return self._run_aiter(max_steps, includes)

    async def run_until_complete(self, user_input: RunnerMessages | str) -> list[AgentChunk]:
        """Run the agent until it completes and return the final message."""
        resp = self.run_stream(user_input, includes=["final_message", "usage", "tool_call", "tool_call_result"])
        return [chunk async for chunk in resp]

    async def _run_aiter(self, max_steps: int, includes: list[AgentChunkType]) -> AsyncGenerator[AgentChunk, None]:
        """Run the agent and return a RunResponse object that can be asynchronously iterated for each chunk."""
        steps = 0
        finish_reason = None
        require_confirm = False
        while finish_reason != "stop" and steps < max_steps and not require_confirm:
            resp = await self.agent.stream_async(self.messages)
            async for chunk in resp:
                if chunk["type"] == "final_message":
                    message = chunk["message"]
                    self.messages.append(message.model_dump())  # type: ignore
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
                if chunk["type"] == "require_confirm":
                    require_confirm = True
            steps += 1
