import litellm
from funcall import Funcall
from litellm.types.utils import ChatCompletionDeltaToolCall, StreamingChoices

from easy_agent.loggers import logger
from easy_agent.types import AssistantMessage, ToolCall, ToolCallFunction


class StreamChunkProcessor:
    """Processor for handling streaming responses"""

    def __init__(self, fc: Funcall) -> None:
        self.fc = fc
        self.current_message: AssistantMessage = None

    def initialize_message(self, chunk: litellm.ModelResponseStream, choice: StreamingChoices) -> None:
        """Initialize the message object"""
        delta = choice.delta
        self.current_message = AssistantMessage(
            id=chunk.id,
            index=choice.index,
            role=delta.role,
            content="",
        )
        logger.debug("Initialized new message: %s", self.current_message.id)

    def update_content(self, content: str) -> None:
        """Update message content"""
        if self.current_message and content:
            self.current_message.content += content

    def _initialize_tool_calls(self, tool_calls: list[litellm.ChatCompletionMessageToolCall]) -> None:
        """Initialize tool calls"""
        if not self.current_message:
            return

        self.current_message.tool_calls = []
        for call in tool_calls:
            logger.debug("Create new tool call: %s", call.id)

    def _update_tool_calls(self, tool_calls: list[litellm.ChatCompletionMessageToolCall]) -> None:
        """Update existing tool calls"""
        if not self.current_message or not self.current_message.tool_calls:
            return

        for current_call, new_call in zip(self.current_message.tool_calls, tool_calls, strict=False):
            if new_call.function.arguments:
                current_call.function.arguments += new_call.function.arguments
            if new_call.type:
                current_call.type = new_call.type

    def update_tool_calls(self, tool_calls: list[ChatCompletionDeltaToolCall]) -> None:
        """Handle tool call updates"""
        if not tool_calls:
            return
        for call in tool_calls:
            if call.id:
                new_tool_call = ToolCall(
                    id=call.id,
                    type=call.type,
                    function=ToolCallFunction(
                        name=call.function.name or "",
                        arguments=call.function.arguments,
                    ),
                    index=call.index,
                )
                if self.current_message.tool_calls is None:
                    self.current_message.tool_calls = []
                self.current_message.tool_calls.append(new_tool_call)
            else:
                existing_call = self.current_message.tool_calls[call.index]
                if call.function.arguments:
                    existing_call.function.arguments += call.function.arguments

    def handle_usage_info(self, chunk: litellm.ModelResponseStream) -> litellm.Usage | None:
        """Handle usage info, return whether this chunk should be skipped"""
        usage = getattr(chunk, "usage", None)
        if usage:
            logger.debug("Model usage: %s", usage)
        return usage

    def finalize_message(self) -> AssistantMessage:
        """Finalize message processing"""
        logger.debug("Message finalized: %s", self.current_message)
        return self.current_message
