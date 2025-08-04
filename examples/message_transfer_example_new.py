"""
Example demonstrating how to use the message_transfer callback feature.

This example shows how to create an agent with a message_transfer callback
that preprocesses messages before they are sent to the API.
"""

import re

from lite_agent.agent import Agent
from lite_agent.message_transfers import consolidate_history_transfer
from lite_agent.types import (
    AssistantTextContent,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    RunnerMessages,
    UserTextContent,
)


def example_message_transfer(messages: RunnerMessages) -> RunnerMessages:
    """Example message transfer callback that adds a prefix to user messages.

    Args:
        messages: The original messages to be processed

    Returns:
        Processed messages with user message content prefixed
    """
    processed_messages = []

    for message in messages:
        if isinstance(message, dict):
            # Handle dict format messages
            if message.get("role") == "user" and "content" in message:
                # Add prefix to user messages
                content = message.get("content", "")
                processed_message = message.copy()
                processed_message["content"] = f"[PROCESSED] {content}"
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        elif isinstance(message, NewUserMessage):
            # Handle user message models
            processed_message = message.model_copy()
            # Process text content items
            for i, content_item in enumerate(processed_message.content):
                if isinstance(content_item, UserTextContent):
                    processed_message.content[i] = UserTextContent(text=f"[PROCESSED] {content_item.text}")
            processed_messages.append(processed_message)
        else:
            # Keep other messages unchanged
            processed_messages.append(message)

    return processed_messages


def privacy_filter_transfer(messages: RunnerMessages) -> RunnerMessages:
    """Example message transfer callback that filters out sensitive information.

    Args:
        messages: The original messages to be processed

    Returns:
        Messages with sensitive information filtered out
    """
    # Simple regex patterns for common sensitive data
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    phone_pattern = r"\b\d{3}-\d{3}-\d{4}\b"
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"

    def filter_content(content: str) -> str:
        """Filter sensitive information from content string."""
        filtered = re.sub(email_pattern, "[EMAIL_REDACTED]", content)
        filtered = re.sub(phone_pattern, "[PHONE_REDACTED]", filtered)
        return re.sub(ssn_pattern, "[SSN_REDACTED]", filtered)

    processed_messages = []

    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str) and "content" in message:
                processed_message = message.copy()
                processed_message["content"] = filter_content(content)
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        elif isinstance(message, NewSystemMessage):
            # Handle system message with string content
            processed_message = message.model_copy()
            processed_message.content = filter_content(message.content)
            processed_messages.append(processed_message)
        elif isinstance(message, NewUserMessage):
            # Handle user message models with content list
            processed_message = message.model_copy()
            # Process text content items
            for i, content_item in enumerate(processed_message.content):
                if isinstance(content_item, UserTextContent):
                    processed_message.content[i] = UserTextContent(text=filter_content(content_item.text))
            processed_messages.append(processed_message)
        elif isinstance(message, NewAssistantMessage):
            # Handle assistant message models with content list
            processed_message = message.model_copy()
            # Process text content items
            for i, content_item in enumerate(processed_message.content):
                if isinstance(content_item, AssistantTextContent):
                    processed_message.content[i] = AssistantTextContent(text=filter_content(content_item.text))
            processed_messages.append(processed_message)
        else:
            # Keep other message types unchanged
            processed_messages.append(message)

    return processed_messages


async def main():
    """Example usage of agents with message transfer callbacks."""

    # Example 1: Agent with message prefix callback
    agent1 = Agent(
        model="gpt-4.1-mini",
        name="PrefixAgent",
        instructions="You are a helpful assistant.",
        message_transfer=example_message_transfer,
    )

    # Example 2: Agent with privacy filter callback
    agent2 = Agent(
        model="gpt-4.1-mini",
        name="PrivacyAgent",
        instructions="You are a privacy-aware assistant.",
        message_transfer=privacy_filter_transfer,
    )

    # Example 3: Agent without callback initially, then set one
    agent3 = Agent(
        model="gpt-4.1-mini",
        name="DynamicAgent",
        instructions="You are a flexible assistant.",
    )

    # Example 4: Agent with predefined history consolidation callback
    agent4 = Agent(
        model="gpt-4.1-mini",
        name="HistoryAgent",
        instructions="You are an assistant that processes conversation history.",
        message_transfer=consolidate_history_transfer,
    )

    # Set callback after initialization
    agent3.set_message_transfer(example_message_transfer)

    print("Agents created successfully with message transfer callbacks!")
    print(f"Agent1 has callback: {agent1.message_transfer is not None}")
    print(f"Agent2 has callback: {agent2.message_transfer is not None}")
    print(f"Agent3 has callback: {agent3.message_transfer is not None}")
    print(f"Agent4 has callback: {agent4.message_transfer is not None}")

    # Example of how the callback would process messages
    test_messages = [
        {"role": "user", "content": "Hi, my email is john@example.com and my phone is 123-456-7890"},
    ]

    print("\nOriginal messages:", test_messages)
    print("After privacy filter:", privacy_filter_transfer(test_messages))
    print("After prefix filter:", example_message_transfer(test_messages))

    # Demonstrate history consolidation with multiple messages
    complex_messages = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'll check the weather for you."},
        {"type": "function_call", "name": "get_weather", "arguments": '{"location": "New York"}'},
        {"type": "function_call_output", "call_id": "call_123", "output": "Sunny, 25°C"},
        {"role": "assistant", "content": "The weather in New York is sunny with a temperature of 25°C."},
        {"role": "user", "content": "What about tomorrow?"},
    ]

    print("\nComplex conversation history:")
    consolidated = consolidate_history_transfer(complex_messages)
    print("Consolidated result:")
    if consolidated and isinstance(consolidated[0], dict) and "content" in consolidated[0]:
        print(consolidated[0]["content"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
