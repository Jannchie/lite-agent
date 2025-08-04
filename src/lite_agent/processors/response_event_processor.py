from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from aiofiles.threadpool.text import AsyncTextIOWrapper
from litellm.types.llms.openai import (
    ContentPartAddedEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)

from lite_agent.types import (
    AgentChunk,
    AssistantMessageEvent,
    AssistantMessageMeta,
    ContentDeltaEvent,
    EventUsage,
    FunctionCallEvent,
    NewAssistantMessage,
    ResponseRawEvent,
    Timing,
    TimingEvent,
    UsageEvent,
)


class ResponseEventProcessor:
    """Processor for handling response events"""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._start_time: datetime | None = None
        self._first_output_time: datetime | None = None
        self._output_complete_time: datetime | None = None
        self._usage_time: datetime | None = None
        self._usage_data: dict[str, Any] = {}

    async def process_chunk(
        self,
        chunk: ResponsesAPIStreamingResponse,
        record_file: AsyncTextIOWrapper | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        # Mark start time on first chunk
        if self._start_time is None:
            self._start_time = datetime.now(timezone.utc)

        if record_file:
            await record_file.write(chunk.model_dump_json() + "\n")
            await record_file.flush()

        yield ResponseRawEvent(raw=chunk)

        events = self.handle_event(chunk)
        for event in events:
            yield event

    def handle_event(self, event: ResponsesAPIStreamingResponse) -> list[AgentChunk]:  # noqa: PLR0911
        """Handle individual response events"""
        if event.type in (
            ResponsesAPIStreamEvents.RESPONSE_CREATED,
            ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ):
            return []

        if isinstance(event, OutputItemAddedEvent):
            self._messages.append(event.item)  # type: ignore
            return []

        if isinstance(event, ContentPartAddedEvent):
            latest_message = self._messages[-1] if self._messages else None
            if latest_message and isinstance(latest_message.get("content"), list):
                latest_message["content"].append(event.part)
            return []

        if isinstance(event, OutputTextDeltaEvent):
            # Mark first output time if not already set
            if self._first_output_time is None:
                self._first_output_time = datetime.now(timezone.utc)

            latest_message = self._messages[-1] if self._messages else None
            if latest_message and isinstance(latest_message.get("content"), list):
                latest_content = latest_message["content"][-1]
                if "text" in latest_content:
                    latest_content["text"] += event.delta
                    return [ContentDeltaEvent(delta=event.delta)]
            return []

        if isinstance(event, OutputItemDoneEvent):
            item = event.item
            if item.get("type") == "function_call":
                return [
                    FunctionCallEvent(
                        call_id=item["call_id"],
                        name=item["name"],
                        arguments=item["arguments"],
                    ),
                ]
            if item.get("type") == "message":
                # Mark output complete time when message is done
                if self._output_complete_time is None:
                    self._output_complete_time = datetime.now(timezone.utc)

                content = item.get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    end_time = datetime.now(timezone.utc)
                    latency_ms = None
                    output_time_ms = None
                    # latency_ms: 从开始准备输出到 LLM 输出第一个字符的时间差
                    if self._start_time and self._first_output_time:
                        latency_ms = int((self._first_output_time - self._start_time).total_seconds() * 1000)
                    # output_time_ms: 从输出第一个字符到输出完成的时间差
                    if self._first_output_time and self._output_complete_time:
                        output_time_ms = int((self._output_complete_time - self._first_output_time).total_seconds() * 1000)

                    meta = AssistantMessageMeta(
                        sent_at=end_time,
                        latency_ms=latency_ms,
                        output_time_ms=output_time_ms,
                        input_tokens=self._usage_data.get("input_tokens"),
                        output_tokens=self._usage_data.get("output_tokens"),
                    )
                    return [
                        AssistantMessageEvent(
                            message=NewAssistantMessage(content=[], meta=meta),
                        ),
                    ]

        elif isinstance(event, FunctionCallArgumentsDeltaEvent):
            if self._messages:
                latest_message = self._messages[-1]
                if latest_message.get("type") == "function_call":
                    if "arguments" not in latest_message:
                        latest_message["arguments"] = ""
                    latest_message["arguments"] += event.delta
            return []

        elif isinstance(event, FunctionCallArgumentsDoneEvent):
            if self._messages:
                latest_message = self._messages[-1]
                if latest_message.get("type") == "function_call":
                    latest_message["arguments"] = event.arguments
            return []

        elif isinstance(event, ResponseCompletedEvent):
            usage = event.response.usage
            if usage:
                # Mark usage time
                self._usage_time = datetime.now(timezone.utc)
                # Store usage data for meta information
                self._usage_data["input_tokens"] = usage.input_tokens
                self._usage_data["output_tokens"] = usage.output_tokens
                # Also store usage time for later calculation
                self._usage_data["usage_time"] = self._usage_time

                results = []

                # First yield usage event
                results.append(
                    UsageEvent(
                        usage=EventUsage(
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                        ),
                    ),
                )

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

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Get the accumulated messages"""
        return self._messages

    def reset(self) -> None:
        """Reset the processor state"""
        self._messages = []
        self._start_time = None
        self._first_output_time = None
        self._output_complete_time = None
        self._usage_time = None
        self._usage_data = {}
