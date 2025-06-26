from collections.abc import AsyncGenerator, Callable

import litellm
from funcall import Funcall
from litellm import CustomStreamWrapper

from lite_agent.loggers import logger
from lite_agent.stream_handlers import litellm_stream_handler
from lite_agent.types import AgentChunk, AgentMessage, RunnerMessages, ToolCall, ToolCallChunk, ToolCallResultChunk


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
            tool_choice="auto",  # TODO: make this configurable
            stream=True,
        )

        # Ensure resp is a CustomStreamWrapper
        if isinstance(resp, CustomStreamWrapper):
            return litellm_stream_handler(resp)
        msg = "Response is not a CustomStreamWrapper, cannot stream chunks."
        raise TypeError(msg)

    async def list_require_confirm_tools(self, tool_calls: list[ToolCall] | None) -> list[ToolCall]:
        if not tool_calls:
            return []
        results = []
        for tool_call in tool_calls:
            tool_func = self.fc.function_registry.get(tool_call.function.name)
            if not tool_func:
                logger.warning("Tool function %s not found in registry", tool_call.function.name)
                continue
            tool_meta = self.fc.get_tool_meta(tool_call.function.name)
            if tool_meta["require_confirm"]:
                logger.debug("Tool call %s requires confirmation", tool_call.id)
                results.append(tool_call)
        return results

    async def handle_tool_calls(self, tool_calls: list[ToolCall] | None) -> AsyncGenerator[ToolCallChunk | ToolCallResultChunk, None]:
        if not tool_calls:
            return
        if tool_calls:
            for tool_call in tool_calls:
                tool_func = self.fc.function_registry.get(tool_call.function.name)
                if not tool_func:
                    logger.warning("Tool function %s not found in registry", tool_call.function.name)
                    continue

            for tool_call in tool_calls:
                try:
                    yield ToolCallChunk(
                        type="tool_call",
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments or "",
                    )
                    content = await self.fc.call_function_async(tool_call.function.name, tool_call.function.arguments or "")
                    yield ToolCallResultChunk(
                        type="tool_call_result",
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        content=str(content),
                    )
                except Exception as e:  # noqa: PERF203
                    logger.exception("Tool call %s failed", tool_call.id)
                    yield ToolCallResultChunk(
                        type="tool_call_result",
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        content=str(e),
                    )
