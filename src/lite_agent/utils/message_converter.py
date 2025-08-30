"""Message format converters for API compatibility."""

from typing import Any

from lite_agent.loggers import logger
from lite_agent.types import (
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    RunnerMessages,
    message_to_llm_dict,
)


class MessageFormatConverter:
    """Converter for different message API formats."""

    @staticmethod
    def to_completion_format(messages: RunnerMessages) -> list[dict]:
        """Convert messages to completion API format.

        This method replaces the complex _convert_responses_to_completions_format
        with a cleaner, more maintainable implementation.
        """
        logger.debug(f"Converting {len(messages)} messages to completion format")
        converted_messages = []

        for message in messages:
            if isinstance(message, (NewUserMessage, NewSystemMessage)):
                # Handle user and system messages directly
                converted_msg = message_to_llm_dict(message)
                if isinstance(message, NewUserMessage):
                    converted_msg = MessageFormatConverter._convert_user_content(converted_msg)
                converted_messages.append(converted_msg)

            elif isinstance(message, NewAssistantMessage):
                # Handle assistant messages with tool calls
                assistant_msg, tool_results = MessageFormatConverter._process_assistant_message(message)
                converted_messages.append(assistant_msg)
                converted_messages.extend(tool_results)

            elif isinstance(message, dict):
                # Handle legacy dict messages with fallback
                converted_msg = MessageFormatConverter._handle_legacy_dict_message(message)
                if converted_msg:
                    converted_messages.extend(converted_msg if isinstance(converted_msg, list) else [converted_msg])

        logger.debug(f"Completed conversion: {len(messages)} -> {len(converted_messages)} messages")
        return converted_messages

    @staticmethod
    def _process_assistant_message(message: NewAssistantMessage) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Process assistant message and extract tool calls/results."""
        text_parts = []
        tool_calls = []
        tool_results = []

        for content_item in message.content:
            if content_item.type == "text":
                text_parts.append(content_item.text)
            elif content_item.type == "tool_call":
                tool_calls.append({
                    "id": content_item.call_id,
                    "type": "function",
                    "function": {
                        "name": content_item.name,
                        "arguments": content_item.arguments if isinstance(content_item.arguments, str) else str(content_item.arguments),
                    },
                    "index": len(tool_calls),
                })
            elif content_item.type == "tool_call_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": content_item.call_id,
                    "content": content_item.output,
                })

        # Create assistant message
        assistant_msg = {
            "role": "assistant",
            "content": " ".join(text_parts) if text_parts else None,
        }

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        return assistant_msg, tool_results

    @staticmethod
    def _convert_user_content(message_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert user message content for completion API."""
        content = message_dict.get("content")
        if not isinstance(content, list):
            return message_dict

        converted_content = []
        for item in content:
            # Handle both Pydantic objects and dicts
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = item
            else:
                converted_content.append(item)
                continue

            item_type = item_dict.get("type")
            if item_type in ["input_text", "text"]:
                converted_content.append({
                    "type": "text",
                    "text": item_dict["text"],
                })
            elif item_type in ["input_image", "image"]:
                if item_dict.get("file_id"):
                    logger.warning("File ID input not supported for Completion API, skipping")
                    continue

                if not item_dict.get("image_url"):
                    logger.warning("Image content missing image_url, skipping")
                    continue

                image_data = {"url": item_dict["image_url"]}
                detail = item_dict.get("detail", "auto")
                if detail:
                    image_data["detail"] = detail

                converted_content.append({
                    "type": "image_url",
                    "image_url": image_data,
                })
            else:
                # Keep other formats as-is
                converted_content.append(item_dict)

        result = message_dict.copy()
        result["content"] = converted_content
        return result

    @staticmethod
    def _handle_legacy_dict_message(message: dict) -> dict | list[dict] | None:
        """Handle legacy dict message formats with simplified logic."""
        message_type = message.get("type")
        role = message.get("role")

        if message_type == "function_call_output":
            return {
                "role": "tool",
                "tool_call_id": message.get("call_id", ""),
                "content": message.get("output", ""),
            }
        if message_type == "function_call":
            # Function calls should be handled as part of assistant messages
            logger.debug("Standalone function_call message encountered, may be processed with assistant message")
            return None
        if role in ["user", "system", "assistant"]:
            # Standard message, convert content if needed
            converted_msg = message.copy()
            if role == "user" and isinstance(message.get("content"), list):
                converted_msg = MessageFormatConverter._convert_user_content(converted_msg)
            return converted_msg
        logger.warning(f"Unknown message format: {message}")
        return None


class ResponsesFormatConverter:
    """Converter for responses API format."""

    @staticmethod
    def to_responses_format(messages: RunnerMessages) -> list[dict[str, Any]]:
        """Convert messages to responses API format."""
        result = []

        for message in messages:
            if isinstance(message, NewAssistantMessage):
                # Convert assistant message content directly
                contents = []
                for item in message.content:
                    if isinstance(item, AssistantTextContent):
                        contents.append({
                            "role": "assistant",
                            "content": item.text,
                        })
                    elif isinstance(item, AssistantToolCall):
                        contents.append({
                            "type": "function_call",
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        })
                    elif isinstance(item, AssistantToolCallResult):
                        contents.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": item.output,
                        })
                result.extend(contents)

            elif isinstance(message, NewUserMessage):
                contents = []
                for item in message.content:
                    match item.type:
                        case "text":
                            contents.append({
                                "type": "input_text",
                                "text": item.text,
                            })
                        case "image":
                            contents.append({
                                "type": "input_image",
                                "image_url": item.image_url,
                            })
                        case "file":
                            contents.append({
                                "type": "input_file",
                                "file_id": item.file_id,
                                "file_name": item.file_name,
                            })

                result.append({
                    "role": message.role,
                    "content": contents,
                })

            elif isinstance(message, NewSystemMessage):
                result.append({
                    "role": "system",
                    "content": message.content,
                })

        return result
