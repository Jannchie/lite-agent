from collections.abc import AsyncGenerator

import litellm
from funcall import Funcall
from funcall.types import Context
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

from lite_agent.loggers import logger
from lite_agent.processors import StreamChunkProcessor
from lite_agent.types import AgentChunk, AssistantMessage, ContentDeltaChunk, FinalMessageChunk, LiteLLMRawChunk, ToolCallChunk, ToolCallDeltaChunk, ToolCallResultChunk, UsageChunk


async def handle_usage_chunk(processor: StreamChunkProcessor, chunk: ModelResponseStream) -> UsageChunk | None:
    usage = processor.handle_usage_info(chunk)
    if usage:
        return UsageChunk(type="usage", usage=usage)
    return None


async def handle_content_and_tool_calls(
    processor: StreamChunkProcessor,
    chunk: ModelResponseStream,
    choice: StreamingChoices,
    delta: Delta,
) -> list[AgentChunk]:
    results: list[AgentChunk] = []
    if not processor.is_initialized:
        processor.initialize_message(chunk, choice)
    if delta.content:
        results.append(ContentDeltaChunk(type="content_delta", delta=delta.content))
        processor.update_content(delta.content)
    if delta.tool_calls is not None:
        processor.update_tool_calls(delta.tool_calls)
        if delta.tool_calls and processor.current_message.tool_calls:
            results.extend(
                [
                    ToolCallDeltaChunk(
                        type="tool_call_delta",
                        tool_call_id=processor.current_message.tool_calls[-1].id,
                        name=processor.current_message.tool_calls[-1].function.name,
                        arguments_delta=tool_call.function.arguments or "",
                    )
                    for tool_call in delta.tool_calls
                    if tool_call.function.arguments
                ],
            )
    return results


async def handle_tool_calls(
    message: AssistantMessage,
    funcall: Funcall,
    context: Context | None = None,
) -> list[AgentChunk]:
    results: list[AgentChunk] = []
    tool_calls = message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            tool_func = funcall.function_registry.get(tool_call.function.name)
            if not tool_func:
                logger.warning("Tool function %s not found in registry", tool_call.function.name)
                continue

        for tool_call in tool_calls:
            try:
                results.append(
                    ToolCallChunk(
                        type="tool_call",
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments or "",
                    ),
                )
                content = await funcall.call_function_async(tool_call.function.name, tool_call.function.arguments or "", context=context)
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


async def litellm_stream_handler(
    resp: litellm.CustomStreamWrapper,
) -> AsyncGenerator[AgentChunk, None]:
    """
    Optimized chunk handler (refactored for simplicity)
    """
    processor = StreamChunkProcessor()
    async for chunk in resp:  # type: ignore
        yield LiteLLMRawChunk(type="litellm_raw", raw=chunk)

        if not isinstance(chunk, ModelResponseStream):
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
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        for result in await handle_content_and_tool_calls(processor, chunk, choice, delta):
            yield result
        # Check if finished
        if choice.finish_reason:
            current_message = processor.current_message
            yield FinalMessageChunk(type="final_message", message=current_message, finish_reason=choice.finish_reason)
