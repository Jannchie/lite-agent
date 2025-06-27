from collections.abc import AsyncGenerator, Callable, Sequence
from pathlib import Path

import litellm
from funcall import Funcall
from litellm import CustomStreamWrapper
from pydantic import BaseModel

from lite_agent.loggers import logger
from lite_agent.stream_handlers import litellm_stream_handler
from lite_agent.types import AgentChunk, AgentSystemMessage, RunnerMessages, ToolCall, ToolCallChunk, ToolCallResultChunk


class Agent:
    def __init__(self, *, model: str, name: str, instructions: str, tools: list[Callable] | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.fc = Funcall(tools)
        self.model = model

    def prepare_messages(self, messages: RunnerMessages) -> list[dict[str, str]]:
        # Convert from responses format to completions format
        converted_messages = self._convert_responses_to_completions_format(messages)

        return [
            AgentSystemMessage(
                role="system",
                content=f"You are {self.name}. {self.instructions}",
            ).model_dump(),
            *converted_messages,
        ]

    async def stream_async(self, messages: RunnerMessages, record_to_file: Path | None = None) -> AsyncGenerator[AgentChunk, None]:
        self.message_histories = self.prepare_messages(messages)
        tools = self.fc.get_tools(target="completion")
        resp = await litellm.acompletion(
            model=self.model,
            messages=self.message_histories,
            tools=tools,
            tool_choice="auto",  # TODO: make this configurable
            stream=True,
        )

        # Ensure resp is a CustomStreamWrapper
        if isinstance(resp, CustomStreamWrapper):
            return litellm_stream_handler(resp, record_to=record_to_file)
        msg = "Response is not a CustomStreamWrapper, cannot stream chunks."
        raise TypeError(msg)

    async def list_require_confirm_tools(self, tool_calls: Sequence[ToolCall] | None) -> Sequence[ToolCall]:
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
                logger.debug('Tool call "%s" requires confirmation', tool_call.id)
                results.append(tool_call)
        return results

    async def handle_tool_calls(self, tool_calls: Sequence[ToolCall] | None) -> AsyncGenerator[ToolCallChunk | ToolCallResultChunk, None]:
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

    def _convert_responses_to_completions_format(self, messages: RunnerMessages) -> list[dict]:
        """Convert messages from responses API format to completions API format."""
        converted_messages = []
        i = 0

        while i < len(messages):
            message = messages[i]
            message_dict = message.model_dump() if isinstance(message, BaseModel) else message

            message_type = message_dict.get("type")
            role = message_dict.get("role")

            if role == "assistant":
                # Look ahead for function_call messages
                tool_calls = []
                j = i + 1

                while j < len(messages):
                    next_message = messages[j]
                    next_dict = next_message.model_dump() if isinstance(next_message, BaseModel) else next_message

                    if next_dict.get("type") == "function_call":
                        tool_call = {
                            "id": next_dict["function_call_id"],
                            "type": "function",
                            "function": {
                                "name": next_dict["name"],
                                "arguments": next_dict["arguments"],
                            },
                            "index": len(tool_calls),
                        }
                        tool_calls.append(tool_call)
                        j += 1
                    else:
                        break

                # Create assistant message with tool_calls if any
                assistant_msg = message_dict.copy()
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                converted_messages.append(assistant_msg)
                i = j  # Skip the function_call messages we've processed

            elif message_type == "function_call_output":
                # Convert to tool message
                converted_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message_dict["call_id"],
                        "content": message_dict["output"],
                    },
                )
                i += 1

            elif message_type == "function_call":
                # This should have been processed with the assistant message
                # Skip it if we encounter it standalone
                i += 1

            else:
                # Regular message (user, system)
                converted_messages.append(message_dict)
                i += 1

        return converted_messages
