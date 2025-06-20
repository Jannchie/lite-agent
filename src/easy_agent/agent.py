from collections.abc import AsyncGenerator, Callable

import litellm
from funcall import Funcall

from easy_agent.chunk_handler import AgentChunk, chunk_handler
from easy_agent.types import RunnerMessages


class Agent:
    def __init__(self, *, model: str, name: str, instructions: str, tools: list[Callable] | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.fc = Funcall(tools)
        self.model = model

    def prepare_messages(self, messages: RunnerMessages) -> list[dict]:
        return [
            {
                "role": "system",
                "content": f"You are {self.name}. {self.instructions}",
            },
            *messages,
        ]

    async def stream_async(self, messages: RunnerMessages) -> AsyncGenerator[AgentChunk, None]:
        self.message_histories = self.prepare_messages(messages)
        tools = self.fc.get_tools(target="litellm")
        resp = await litellm.acompletion(
            model=self.model,
            messages=self.message_histories,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )
        return chunk_handler(resp, self.fc)
