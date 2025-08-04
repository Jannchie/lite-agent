"""
Predefined message transfer functions for lite-agent.

This module provides common message transfer functions that can be used
with agents to preprocess messages before sending them to the API.
"""

from lite_agent.types import RunnerMessages


def consolidate_history_transfer(messages: RunnerMessages) -> RunnerMessages:
    """Consolidate all message history into a single user message with XML format.

    This message transfer function converts all message history into XML format
    and creates a single user message asking what to do next. This is useful when
    you want to summarize the entire conversation context in a single prompt.

    Args:
        messages: The original messages to be processed

    Returns:
        A single user message containing the consolidated history in XML format

    Example:
        >>> agent = Agent(
        ...     model="gpt-4",
        ...     name="HistoryAgent",
        ...     instructions="You are a helpful assistant.",
        ...     message_transfer=consolidate_history_transfer
        ... )
    """
    if not messages:
        return messages

    # Convert messages to XML format
    xml_content = ["<conversation_history>"]

    for message in messages:
        xml_content.extend(_process_message_to_xml(message))

    xml_content.append("</conversation_history>")

    # Create the consolidated message
    consolidated_content = "以下是目前发生的所有交互:\n\n" + "\n".join(xml_content) + "\n\n接下来该做什么?"

    # Return a single user message
    return [{"role": "user", "content": consolidated_content}]


def _process_message_to_xml(message: dict | object) -> list[str]:
    """Process a single message and convert it to XML format.

    Args:
        message: A single message to process

    Returns:
        List of XML strings representing the message
    """
    xml_lines = []

    if isinstance(message, dict):
        xml_lines.extend(_process_dict_message(message))
    elif hasattr(message, "role"):
        # Handle Pydantic model format messages
        role = getattr(message, "role", "unknown")
        content = getattr(message, "content", "")

        # Handle new message format where content is a list
        if isinstance(content, list):
            # Extract text from content items
            text_parts = [item.text for item in content if (hasattr(item, "type") and item.type == "text") or hasattr(item, "text")]
            content_text = " ".join(text_parts)
            if content_text:
                xml_lines.append(f"  <message role='{role}'>{content_text}</message>")
        elif isinstance(content, str):
            xml_lines.append(f"  <message role='{role}'>{content}</message>")
    elif hasattr(message, "type"):
        # Handle function call messages
        xml_lines.extend(_process_function_message(message))

    return xml_lines


def _process_dict_message(message: dict) -> list[str]:
    """Process dictionary format message to XML."""
    xml_lines = []
    role = message.get("role", "unknown")
    content = message.get("content", "")
    message_type = message.get("type")

    if message_type == "function_call":
        name = message.get("name", "unknown")
        arguments = message.get("arguments", "")
        xml_lines.append(f"  <function_call name='{name}' arguments='{arguments}' />")
    elif message_type == "function_call_output":
        call_id = message.get("call_id", "unknown")
        output = message.get("output", "")
        xml_lines.append(f"  <function_result call_id='{call_id}'>{output}</function_result>")
    elif role in ["user", "assistant", "system"]:
        xml_lines.append(f"  <message role='{role}'>{content}</message>")

    return xml_lines


def _process_function_message(message: dict | object) -> list[str]:
    """Process function call message to XML."""
    xml_lines = []
    message_type = getattr(message, "type", "unknown")

    if message_type == "function_call":
        name = getattr(message, "name", "unknown")
        arguments = getattr(message, "arguments", "")
        xml_lines.append(f"  <function_call name='{name}' arguments='{arguments}' />")
    elif message_type == "function_call_output":
        call_id = getattr(message, "call_id", "unknown")
        output = getattr(message, "output", "")
        xml_lines.append(f"  <function_result call_id='{call_id}'>{output}</function_result>")

    return xml_lines
