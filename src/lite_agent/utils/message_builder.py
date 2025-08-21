import json
from typing import Any

from lite_agent.types import (
    AssistantMessageContent,
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    MessageMeta,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    UserImageContent,
    UserMessageContent,
    UserTextContent,
)


class MessageBuilder:
    """Utility class for building and converting messages from various formats."""

    @staticmethod
    def build_user_message_from_dict(message: dict[str, Any]) -> NewUserMessage:
        """Build a NewUserMessage from a dictionary.

        Args:
            message: Dictionary containing user message data

        Returns:
            NewUserMessage instance
        """
        content = message.get("content", "")

        # Preserve meta information if present
        meta_data = message.get("meta", {})
        meta = MessageMeta(**meta_data) if meta_data else MessageMeta()

        if isinstance(content, str):
            return NewUserMessage(content=[UserTextContent(text=content)], meta=meta)

        if isinstance(content, list):
            return NewUserMessage(content=MessageBuilder._build_user_content_items(content), meta=meta)

        # Handle non-string, non-list content
        return NewUserMessage(content=[UserTextContent(text=str(content))], meta=meta)

    @staticmethod
    def _build_user_content_items(content_list: list[Any]) -> list[UserMessageContent]:
        """Build user content items from a list of content data.

        Args:
            content_list: List of content items (dicts or objects)

        Returns:
            List of UserMessageContent items
        """
        user_content_items: list[UserMessageContent] = []

        for item in content_list:
            if isinstance(item, dict):
                user_content_items.append(MessageBuilder._build_user_content_from_dict(item))
            elif hasattr(item, "type"):
                user_content_items.append(MessageBuilder._build_user_content_from_object(item))
            else:
                # Fallback: convert to text
                user_content_items.append(UserTextContent(text=str(item)))

        return user_content_items

    @staticmethod
    def _build_user_content_from_dict(item: dict[str, Any]) -> UserMessageContent:
        """Build user content from a dictionary item.

        Args:
            item: Dictionary containing content item data

        Returns:
            UserMessageContent instance
        """
        item_type = item.get("type")

        if item_type in {"input_text", "text"}:
            return UserTextContent(text=item.get("text", ""))

        if item_type in {"input_image", "image_url"}:
            if item_type == "image_url":
                # Handle completion API format
                image_url_data = item.get("image_url", {})
                url = image_url_data.get("url", "") if isinstance(image_url_data, dict) else str(image_url_data)
                return UserImageContent(image_url=url)

            # Handle response API format
            return UserImageContent(
                image_url=item.get("image_url"),
                file_id=item.get("file_id"),
                detail=item.get("detail", "auto"),
            )

        # Fallback: treat as text
        return UserTextContent(text=str(item.get("text", item)))

    @staticmethod
    def _build_user_content_from_object(item: Any) -> UserMessageContent:  # noqa: ANN401
        """Build user content from an object with attributes.

        Args:
            item: Object with type attribute and other properties

        Returns:
            UserMessageContent instance
        """
        if item.type == "input_text":
            return UserTextContent(text=item.text)

        if item.type == "input_image":
            return UserImageContent(
                image_url=getattr(item, "image_url", None),
                file_id=getattr(item, "file_id", None),
                detail=getattr(item, "detail", "auto"),
            )

        # Fallback: convert to text
        return UserTextContent(text=str(item))

    @staticmethod
    def build_system_message_from_dict(message: dict[str, Any]) -> NewSystemMessage:
        """Build a NewSystemMessage from a dictionary.

        Args:
            message: Dictionary containing system message data

        Returns:
            NewSystemMessage instance
        """
        content = message.get("content", "")

        # Preserve meta information if present
        meta_data = message.get("meta", {})
        if meta_data:
            meta = MessageMeta(**meta_data)
        else:
            meta = MessageMeta()

        return NewSystemMessage(content=str(content), meta=meta)

    @staticmethod
    def build_assistant_message_from_dict(message: dict[str, Any]) -> NewAssistantMessage:
        """Build a NewAssistantMessage from a dictionary.

        Args:
            message: Dictionary containing assistant message data

        Returns:
            NewAssistantMessage instance
        """
        content = message.get("content", "")
        assistant_content_items: list[AssistantMessageContent] = []

        if content:
            if isinstance(content, str):
                assistant_content_items = [AssistantTextContent(text=content)]
            elif isinstance(content, list):
                # Handle array content (from new format messages)
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            assistant_content_items.append(AssistantTextContent(text=item.get("text", "")))
                        elif item_type == "tool_call":
                            assistant_content_items.append(
                                AssistantToolCall(
                                    call_id=item.get("call_id", ""),
                                    name=item.get("name", ""),
                                    arguments=item.get("arguments", "{}"),
                                ),
                            )
                        elif item_type == "tool_call_result":
                            assistant_content_items.append(
                                AssistantToolCallResult(
                                    call_id=item.get("call_id", ""),
                                    output=item.get("output", ""),
                                    execution_time_ms=item.get("execution_time_ms"),
                                ),
                            )
                        # Add more content types as needed
                    else:
                        # Fallback for unknown item format
                        assistant_content_items.append(AssistantTextContent(text=str(item)))
            else:
                # Fallback for other content types
                assistant_content_items = [AssistantTextContent(text=str(content))]

        # Handle tool calls if present (legacy format)
        if "tool_calls" in message:
            for tool_call in message.get("tool_calls", []):
                try:
                    arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                except (json.JSONDecodeError, TypeError):
                    arguments = tool_call["function"]["arguments"]

                assistant_content_items.append(
                    AssistantToolCall(
                        call_id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        arguments=arguments,
                    ),
                )

        # Preserve meta information if present
        meta_data = message.get("meta", {})
        if meta_data:
            meta = AssistantMessageMeta(**meta_data)
        else:
            meta = AssistantMessageMeta()

        return NewAssistantMessage(content=assistant_content_items, meta=meta)
