from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any, TypeAlias, cast

from aiofiles.threadpool.text import AsyncTextIOWrapper
from openai.types.responses import ResponseStreamEvent

from lite_agent.types import (
    AgentChunk,
    AssistantMessageEvent,
    AssistantMessageMeta,
    ContentDeltaEvent,
    EventUsage,
    FunctionCallEvent,
    MessageUsage,
    NewAssistantMessage,
    ResponseRawEvent,
    Timing,
    TimingEvent,
    UsageEvent,
)
from lite_agent.utils.metrics import TimingMetrics

JSONValue: TypeAlias = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None


class ResponseEventProcessor:
    """Processor for handling response events"""

    def __init__(self) -> None:
        self._messages: list[dict[str, JSONValue]] = []
        self._start_time: datetime | None = None
        self._first_output_time: datetime | None = None
        self._output_complete_time: datetime | None = None
        self._usage_time: datetime | None = None
        self._usage_data: dict[str, Any] = {}

    async def process_chunk(
        self,
        chunk: ResponseStreamEvent,
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

    def handle_event(self, event: ResponseStreamEvent) -> list[AgentChunk]:  # noqa: PLR0911
        """Handle individual response events"""
        event_type = getattr(event, "type", None)

        if event_type == "response.output_item.added":
            self._messages.append(cast("dict[str, JSONValue]", self._convert_model(event.item)))
            return []

        if event_type == "response.content_part.added":
            latest_message = self._messages[-1] if self._messages else None
            content = latest_message.get("content") if latest_message else None
            if isinstance(content, list):
                content.append(self._convert_model(event.part))
            return []

        if event_type == "response.output_text.delta":
            # Mark first output time if not already set
            if self._first_output_time is None:
                self._first_output_time = datetime.now(timezone.utc)

            latest_message = self._messages[-1] if self._messages else None
            if latest_message:
                content = latest_message.get("content")
                if isinstance(content, list) and content:
                    latest_content = content[-1]
                    if isinstance(latest_content, dict) and isinstance(latest_content.get("text"), str):
                        delta_text = cast("str", event.delta)
                        latest_content["text"] = f"{latest_content['text']}{delta_text}"
                        return [ContentDeltaEvent(delta=event.delta)]
            return []

        if event_type == "response.output_item.done":
            item = cast("dict[str, JSONValue]", self._convert_model(event.item))
            if item.get("type") == "function_call":
                function_event: AgentChunk = FunctionCallEvent(
                    call_id=cast("str", item["call_id"]),
                    name=cast("str", item["name"]),
                    arguments=item["arguments"],
                )
                return [function_event]
            if item.get("type") == "message":
                # Mark output complete time when message is done
                if self._output_complete_time is None:
                    self._output_complete_time = datetime.now(timezone.utc)

                content = item.get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    end_time = datetime.now(timezone.utc)
                    latency_ms = TimingMetrics.calculate_latency_ms(self._start_time, self._first_output_time)
                    output_time_ms = TimingMetrics.calculate_output_time_ms(self._first_output_time, self._output_complete_time)

                    # Extract model information from event
                    model_name = getattr(event, "model", None)
                    # Debug: check if event has model info in different location
                    if hasattr(event, "response"):
                        response = getattr(event, "response", None)
                        if response and hasattr(response, "model"):
                            model_name = getattr(response, "model", None)
                    # Create usage information
                    usage = MessageUsage(
                        input_tokens=self._usage_data.get("input_tokens"),
                        output_tokens=self._usage_data.get("output_tokens"),
                        total_tokens=(self._usage_data.get("input_tokens") or 0) + (self._usage_data.get("output_tokens") or 0),
                    )
                    meta = AssistantMessageMeta(
                        sent_at=end_time,
                        model=model_name,
                        latency_ms=latency_ms,
                        output_time_ms=output_time_ms,
                        usage=usage,
                    )
                    return [
                        AssistantMessageEvent(
                            message=NewAssistantMessage(content=[], meta=meta),
                        ),
                    ]

        elif event_type == "response.function_call.arguments.delta":
            if self._messages:
                latest_message = self._messages[-1]
                if latest_message.get("type") == "function_call":
                    arguments = latest_message.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = ""
                    latest_message["arguments"] = f"{arguments}{event.delta}"
            return []

        elif event_type == "response.function_call.arguments.done":
            if self._messages:
                latest_message = self._messages[-1]
                if latest_message.get("type") == "function_call":
                    latest_message["arguments"] = event.arguments
            return []

        elif event_type == "response.completed":
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
                latency_ms = TimingMetrics.calculate_latency_ms(self._start_time, self._first_output_time)
                output_time_ms = TimingMetrics.calculate_output_time_ms(self._first_output_time, self._output_complete_time)
                if latency_ms is not None and output_time_ms is not None:
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

    @staticmethod
    def _convert_model(value: object) -> JSONValue:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, list):
            return [ResponseEventProcessor._convert_model(item) for item in value]
        if isinstance(value, dict):
            return {key: ResponseEventProcessor._convert_model(item) for key, item in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return cast("JSONValue", value)
