from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Literal

import litellm
from aiofiles.threadpool.text import AsyncTextIOWrapper
from litellm.types.utils import ChatCompletionDeltaToolCall, ModelResponseStream, StreamingChoices

from lite_agent.loggers import logger
from lite_agent.types import (
    AgentChunk,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageMeta,
    AssistantTextContent,
    CompletionRawEvent,
    ContentDeltaEvent,
    EventUsage,
    FunctionCallDeltaEvent,
    FunctionCallEvent,
    MessageUsage,
    NewAssistantMessage,
    Timing,
    TimingEvent,
    ToolCall,
    ToolCallFunction,
    UsageEvent,
)


class CompletionEventProcessor:
    """Processor for handling completion event"""

    def __init__(self) -> None:
        self._current_message: AssistantMessage | None = None
        self.processing_chunk: Literal["content", "tool_calls"] | None = None
        self.processing_function: str | None = None
        self.last_processed_chunk: ModelResponseStream | None = None
        self.yielded_content = False
        self.yielded_function = set()
        self._start_time: datetime | None = None
        self._first_output_time: datetime | None = None
        self._output_complete_time: datetime | None = None
        self._usage_time: datetime | None = None
        self._usage_data: dict[str, int] = {}

    async def process_chunk(
        self,
        chunk: ModelResponseStream,
        record_file: AsyncTextIOWrapper | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        # Mark start time on first chunk
        if self._start_time is None:
            self._start_time = datetime.now(timezone.utc)

        if record_file:
            await record_file.write(chunk.model_dump_json() + "\n")
            await record_file.flush()
        yield CompletionRawEvent(raw=chunk)
        usage_chunks = self.handle_usage_chunk(chunk)
        if usage_chunks:
            for usage_chunk in usage_chunks:
                yield usage_chunk
            return
        if not chunk.choices:
            return

        choice = chunk.choices[0]
        delta = choice.delta
        if delta.tool_calls:
            if not self.yielded_content:
                self.yielded_content = True
                end_time = datetime.now(timezone.utc)
                latency_ms = None
                output_time_ms = None
                # latency_ms: 从开始准备输出到 LLM 输出第一个字符的时间差
                if self._start_time and self._first_output_time:
                    latency_ms = int((self._first_output_time - self._start_time).total_seconds() * 1000)
                # output_time_ms: 从输出第一个字符到输出完成的时间差
                if self._first_output_time and self._output_complete_time:
                    output_time_ms = int((self._output_complete_time - self._first_output_time).total_seconds() * 1000)

                usage = MessageUsage(
                    input_tokens=self._usage_data.get("input_tokens"),
                    output_tokens=self._usage_data.get("output_tokens"),
                )
                meta = AssistantMessageMeta(
                    sent_at=end_time,
                    latency_ms=latency_ms,
                    total_time_ms=output_time_ms,
                    usage=usage,
                )
                # Include accumulated text content in the message
                content = []
                if self._current_message and self._current_message.content:
                    content.append(AssistantTextContent(text=self._current_message.content))

                yield AssistantMessageEvent(
                    message=NewAssistantMessage(
                        content=content,
                        meta=meta,
                    ),
                )
            first_tool_call = delta.tool_calls[0]
            tool_name = first_tool_call.function.name if first_tool_call.function else ""
            if tool_name:
                self.processing_function = tool_name
        delta = choice.delta
        if (
            self._current_message
            and self._current_message.tool_calls
            and self.processing_function != self._current_message.tool_calls[-1].function.name
            and self._current_message.tool_calls[-1].function.name not in self.yielded_function
        ):
            tool_call = self._current_message.tool_calls[-1]
            yield FunctionCallEvent(
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments or "",
            )
            self.yielded_function.add(tool_call.function.name)
        if not self.is_initialized:
            self.initialize_message(chunk, choice)
        if delta.content and self._current_message:
            # Mark first output time if not already set
            if self._first_output_time is None:
                self._first_output_time = datetime.now(timezone.utc)
            self._current_message.content += delta.content
            yield ContentDeltaEvent(delta=delta.content)
        if delta.tool_calls is not None:
            self.update_tool_calls(delta.tool_calls)
            if delta.tool_calls and self.current_message.tool_calls:
                tool_call = delta.tool_calls[0]
                message_tool_call = self.current_message.tool_calls[-1]
                yield FunctionCallDeltaEvent(
                    tool_call_id=message_tool_call.id,
                    name=message_tool_call.function.name,
                    arguments_delta=tool_call.function.arguments or "",
                )
        if choice.finish_reason:
            # Mark output complete time when finish_reason appears
            if self._output_complete_time is None:
                self._output_complete_time = datetime.now(timezone.utc)

            if self.current_message.tool_calls:
                tool_call = self.current_message.tool_calls[-1]
                yield FunctionCallEvent(
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments or "",
                )
            if not self.yielded_content:
                self.yielded_content = True
                end_time = datetime.now(timezone.utc)
                latency_ms = None
                output_time_ms = None
                # latency_ms: 从开始准备输出到 LLM 输出第一个字符的时间差
                if self._start_time and self._first_output_time:
                    latency_ms = int((self._first_output_time - self._start_time).total_seconds() * 1000)
                # output_time_ms: 从输出第一个字符到输出完成的时间差
                if self._first_output_time and self._output_complete_time:
                    output_time_ms = int((self._output_complete_time - self._first_output_time).total_seconds() * 1000)

                usage = MessageUsage(
                    input_tokens=self._usage_data.get("input_tokens"),
                    output_tokens=self._usage_data.get("output_tokens"),
                )
                meta = AssistantMessageMeta(
                    sent_at=end_time,
                    latency_ms=latency_ms,
                    total_time_ms=output_time_ms,
                    usage=usage,
                )
                # Include accumulated text content in the message
                content = []
                if self._current_message and self._current_message.content:
                    content.append(AssistantTextContent(text=self._current_message.content))

                yield AssistantMessageEvent(
                    message=NewAssistantMessage(
                        content=content,
                        meta=meta,
                    ),
                )
        self.last_processed_chunk = chunk

    def handle_usage_chunk(self, chunk: ModelResponseStream) -> list[AgentChunk]:
        usage = getattr(chunk, "usage", None)
        if usage:
            # Mark usage time
            self._usage_time = datetime.now(timezone.utc)
            # Store usage data for meta information
            self._usage_data["input_tokens"] = usage["prompt_tokens"]
            self._usage_data["output_tokens"] = usage["completion_tokens"]

            results = []

            # First yield usage event
            results.append(UsageEvent(usage=EventUsage(input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"])))

            # Then yield timing event if we have timing data
            if self._start_time and self._first_output_time and self._output_complete_time:
                latency_ms = int((self._first_output_time - self._start_time).total_seconds() * 1000)
                output_time_ms = int((self._output_complete_time - self._first_output_time).total_seconds() * 1000)

                results.append(
                    TimingEvent(
                        timing=Timing(
                            latency_ms=latency_ms,
                            output_time_ms=output_time_ms,
                        ),
                    ),
                )

            return results
        return []

    def initialize_message(self, chunk: ModelResponseStream, choice: StreamingChoices) -> None:
        """Initialize the message object"""
        delta = choice.delta
        if delta.role != "assistant":
            logger.warning("Skipping chunk with role: %s", delta.role)
            return
        self._current_message = AssistantMessage(
            id=chunk.id,
            index=choice.index,
            role=delta.role,
            content="",
        )
        logger.debug('Initialized new message: "%s"', self._current_message.id)

    def update_content(self, content: str) -> None:
        """Update message content"""
        if self._current_message and content:
            self._current_message.content += content

    def _initialize_tool_calls(self, tool_calls: list[litellm.ChatCompletionMessageToolCall]) -> None:
        """Initialize tool calls"""
        if not self._current_message:
            return

        self._current_message.tool_calls = []
        for call in tool_calls:
            logger.debug("Create new tool call: %s", call.id)

    def _update_tool_calls(self, tool_calls: list[litellm.ChatCompletionMessageToolCall]) -> None:
        """Update existing tool calls"""
        if not self._current_message:
            return
        if not hasattr(self._current_message, "tool_calls"):
            self._current_message.tool_calls = []
        if not self._current_message.tool_calls:
            return
        if not tool_calls:
            return
        for current_call, new_call in zip(self._current_message.tool_calls, tool_calls, strict=False):
            if new_call.function.arguments and current_call.function.arguments:
                current_call.function.arguments += new_call.function.arguments
            if new_call.type and new_call.type == "function":
                current_call.type = new_call.type
            elif new_call.type:
                logger.warning("Unexpected tool call type: %s", new_call.type)

    def update_tool_calls(self, tool_calls: list[ChatCompletionDeltaToolCall]) -> None:
        """Handle tool call updates"""
        if not tool_calls:
            return
        for call in tool_calls:
            if call.id:
                if call.type == "function":
                    new_tool_call = ToolCall(
                        id=call.id,
                        type=call.type,
                        function=ToolCallFunction(
                            name=call.function.name or "",
                            arguments=call.function.arguments,
                        ),
                        index=call.index,
                    )
                    if self._current_message is not None:
                        if self._current_message.tool_calls is None:
                            self._current_message.tool_calls = []
                        self._current_message.tool_calls.append(new_tool_call)
                else:
                    logger.warning("Unexpected tool call type: %s", call.type)
            elif self._current_message is not None and self._current_message.tool_calls is not None and call.index is not None and 0 <= call.index < len(self._current_message.tool_calls):
                existing_call = self._current_message.tool_calls[call.index]
                if call.function.arguments:
                    if existing_call.function.arguments is None:
                        existing_call.function.arguments = ""
                    existing_call.function.arguments += call.function.arguments
            else:
                logger.warning("Cannot update tool call: current_message or tool_calls is None, or invalid index.")

    @property
    def is_initialized(self) -> bool:
        """Check if the current message is initialized"""
        return self._current_message is not None

    @property
    def current_message(self) -> AssistantMessage:
        """Get the current message being processed"""
        if not self._current_message:
            msg = "No current message initialized. Call initialize_message first."
            raise ValueError(msg)
        return self._current_message
