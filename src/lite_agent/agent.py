from collections.abc import AsyncGenerator, Callable

import litellm
from funcall import Funcall
from litellm import CustomStreamWrapper

from lite_agent.chunk_handler import AgentChunk, chunk_handler
from lite_agent.types import AgentMessage, RunnerMessages


class Agent:
    def __init__(self, *, model: str, name: str, instructions: str, tools: list[Callable] | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.fc = Funcall(tools)
        self.model = model

    def prepare_messages(self, messages: RunnerMessages) -> list[AgentMessage]:
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
        # Ensure resp is a CustomStreamWrapper
        if isinstance(resp, CustomStreamWrapper):
            return chunk_handler(resp, self.fc)
        msg = "Response is not a CustomStreamWrapper, cannot stream chunks."
        raise TypeError(msg)
