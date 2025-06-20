from collections.abc import AsyncGenerator
from typing import Literal, TypedDict

import litellm
from funcall import Funcall
from litellm.types.utils import Delta, StreamingChoices

from lite_agent.loggers import logger
from lite_agent.processors import StreamChunkProcessor
from lite_agent.processors.stream_chunk_processor import AssistantMessage


class LiteLLMRawChunk(TypedDict):
    """
    Define the type of chunk
    """

    type: Literal["litellm_raw"]
    raw: litellm.ModelResponseStream


class UsageChunk(TypedDict):
    """
    Define the type of usage info chunk
    """

    type: Literal["usage"]
    usage: litellm.Usage


class FinalMessageChunk(TypedDict):
    """
    Define the type of final message chunk
    """

    type: Literal["final_message"]
    message: AssistantMessage
    finish_reason: Literal["stop", "tool_calls"]


class ToolCallChunk(TypedDict):
    """
    Define the type of tool call chunk
    """

    type: Literal["tool_call"]
    name: str
    arguments: str


class ToolCallResultChunk(TypedDict):
    """
    Define the type of tool call result chunk
    """

    type: Literal["tool_call_result"]
    tool_call_id: str
    name: str
    content: str


class ContentDeltaChunk(TypedDict):
    """
    Define the type of message chunk
    """

    type: Literal["content_delta"]
    delta: str


class ToolCallDeltaChunk(TypedDict):
    """
    Define the type of tool call delta chunk
    """

    type: Literal["tool_call_delta"]
    tool_call_id: str
    name: str
    arguments_delta: str


AgentChunk = LiteLLMRawChunk | UsageChunk | FinalMessageChunk | ToolCallChunk | ToolCallResultChunk | ContentDeltaChunk


async def handle_usage_chunk(processor: StreamChunkProcessor, chunk: litellm.ModelResponseStream) -> UsageChunk | None:
    usage = processor.handle_usage_info(chunk)
    if usage:
        return UsageChunk(type="usage", usage=usage)
    return None


async def handle_content_and_tool_calls(
    processor: StreamChunkProcessor,
    chunk: litellm.ModelResponseStream,
    choice: StreamingChoices,
    delta: Delta,
) -> list[AgentChunk]:
    results: list[AgentChunk] = []
    if not processor.current_message:
        processor.initialize_message(chunk, choice)
    if delta.content:
        results.append(ContentDeltaChunk(type="content_delta", delta=delta.content))
    processor.update_content(delta.content)
    processor.update_tool_calls(delta.tool_calls)
    if delta.tool_calls:
        results.extend(
            [
                ToolCallDeltaChunk(
                    type="tool_call_delta",
                    tool_call_id=processor.current_message.tool_calls[-1].id,
                    name=processor.current_message.tool_calls[-1].function.name,
                    arguments_delta=tool_call.function.arguments,
                )
                for tool_call in delta.tool_calls
                if tool_call.function.arguments
            ],
        )
    return results


async def handle_final_message_and_tool_calls(
    processor: StreamChunkProcessor,
    choice: StreamingChoices,
    fc: Funcall,
) -> list[AgentChunk]:
    results: list[AgentChunk] = []
    current_message = processor.finalize_message()
    results.append(FinalMessageChunk(type="final_message", message=current_message, finish_reason=choice.finish_reason))
    tool_calls = current_message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            try:
                results.append(
                    ToolCallChunk(
                        type="tool_call",
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                content = await fc.call_function_async(tool_call.function.name, tool_call.function.arguments)
                results.append(
                    ToolCallResultChunk(
                        type="tool_call_result",
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        content=str(content),
                    ),
                )
            except Exception as e:  # noqa: PERF203
                logger.exception("Tool call %s failed", tool_call.id)
                results.append(
                    ToolCallResultChunk(
                        type="tool_call_result",
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        content=str(e),
                    ),
                )
    return results


async def chunk_handler(
    resp: litellm.CustomStreamWrapper,
    fc: Funcall,
) -> AsyncGenerator[AgentChunk, None]:
    """
    Optimized chunk handler (refactored for simplicity)
    """
    processor = StreamChunkProcessor(fc)
    async for chunk in resp:
        if not isinstance(chunk, litellm.ModelResponseStream):
            logger.debug("unexpected chunk type: %s", type(chunk))
            logger.debug("chunk content: %s", chunk)
            continue

        # Handle usage info
        usage_chunk = await handle_usage_chunk(processor, chunk)
        if usage_chunk:
            yield usage_chunk
            continue

        # Get choice and delta data
        if not chunk.choices:
            yield LiteLLMRawChunk(type="litellm_raw", raw=chunk)
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        for result in await handle_content_and_tool_calls(processor, chunk, choice, delta):
            yield result
        # Check if finished
        if choice.finish_reason and processor.current_message:
            for result in await handle_final_message_and_tool_calls(processor, choice, fc):
                yield result
            continue
        yield LiteLLMRawChunk(type="litellm_raw", raw=chunk)
