import litellm
from litellm.types.utils import ChatCompletionDeltaToolCall, ModelResponseStream, StreamingChoices

from lite_agent.loggers import logger
from lite_agent.types import AssistantMessage, ToolCall, ToolCallFunction


class StreamChunkProcessor:
    """Processor for handling streaming responses"""

    def __init__(self) -> None:
        self._current_message: AssistantMessage | None = None

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
        logger.debug("Initialized new message: %s", self._current_message.id)

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

    def handle_usage_info(self, chunk: ModelResponseStream) -> litellm.Usage | None:
        """Handle usage info, return whether this chunk should be skipped"""
        usage = getattr(chunk, "usage", None)
        if usage:
            logger.debug("Model usage: %s", usage)
        return usage

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
