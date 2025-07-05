from collections.abc import AsyncGenerator
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
    AgentAssistantMessage,
    AgentChunk,
    AssistantMessageEvent,
    ContentDeltaEvent,
    FunctionCallEvent,
    ResponseRawEvent,
    Usage,
    UsageEvent,
)


class ResponseEventProcessor:
    """Processor for handling response events"""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    async def process_chunk(
        self,
        chunk: ResponsesAPIStreamingResponse,
        record_file: AsyncTextIOWrapper | None = None,
    ) -> AsyncGenerator[AgentChunk, None]:
        if record_file:
            await record_file.write(chunk.model_dump_json() + "\n")
            await record_file.flush()

        yield ResponseRawEvent(raw=chunk)

        event = self.handle_event(chunk)
        if event:
            yield event

    def handle_event(self, event: ResponsesAPIStreamingResponse) -> AgentChunk | None:  # noqa: C901, PLR0912
        """Handle individual response events"""
        if event.type in (
            ResponsesAPIStreamEvents.RESPONSE_CREATED,
            ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ):
            return None

        if isinstance(event, OutputItemAddedEvent):
            self._messages.append(event.item)  # type: ignore

        elif isinstance(event, ContentPartAddedEvent):
            latest_message = self._messages[-1] if self._messages else None
            if latest_message and isinstance(latest_message.get("content"), list):
                latest_message["content"].append(event.part)

        elif isinstance(event, OutputTextDeltaEvent):
            latest_message = self._messages[-1] if self._messages else None
            if latest_message and isinstance(latest_message.get("content"), list):
                latest_content = latest_message["content"][-1]
                if "text" in latest_content:
                    latest_content["text"] += event.delta
                    return ContentDeltaEvent(delta=event.delta)

        elif isinstance(event, OutputItemDoneEvent):
            item = event.item
            if item.get("type") == "function_call":
                return FunctionCallEvent(
                    call_id=item["call_id"],
                    name=item["name"],
                    arguments=item["arguments"],
                )
            if item.get("type") == "message":
                content = item.get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get("text", "")
                    return AssistantMessageEvent(
                        message=AgentAssistantMessage(content=text_content),
                    )

        elif isinstance(event, FunctionCallArgumentsDeltaEvent):
            if self._messages:
                latest_message = self._messages[-1]
                if latest_message.get("type") == "function_call":
                    if "arguments" not in latest_message:
                        latest_message["arguments"] = ""
                    latest_message["arguments"] += event.delta

        elif isinstance(event, FunctionCallArgumentsDoneEvent):
            if self._messages:
                latest_message = self._messages[-1]
                if latest_message.get("type") == "function_call":
                    latest_message["arguments"] = event.arguments

        elif isinstance(event, ResponseCompletedEvent):
            usage = event.response.usage
            if usage:
                return UsageEvent(
                    usage=Usage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    ),
                )

        return None

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Get the accumulated messages"""
        return self._messages

    def reset(self) -> None:
        """Reset the processor state"""
        self._messages = []
