#!/usr/bin/env python3
"""
Example demonstrating the new Response API format for mixed text and image content.

This example shows how to use the new ResponseInputText and ResponseInputImage
classes for internal message representation, which are automatically converted
to the legacy Completion API format when making actual LLM calls.
"""

import asyncio

from lite_agent import Agent, Runner
from lite_agent.types import ResponseInputImage, ResponseInputText


async def main():
    """Demonstrate using Response API format for mixed content messages."""

    # Create an agent
    agent = Agent(
        model="gpt-4-turbo",
        name="ImageAnalysisAgent",
        instructions="You are a helpful assistant that can analyze images and answer questions about them.",
    )

    # Create a runner instance
    runner = Runner(agent)

    # Example 1: Traditional text-only message (still works)
    print("=== Example 1: Text-only message ===")
    runner.append_message(
        {
            "role": "user",
            "content": "Hello! Can you help me analyze some images?",
        },
    )

    # Example 2: Using new Response API format with mixed content
    print("\n=== Example 2: Mixed content using Response API format ===")

    # Create message content using new Response API format
    mixed_content = [
        ResponseInputText(
            type="input_text",
            text="What do you think about this photo?",
        ),
        ResponseInputImage(
            type="input_image",
            detail="high",
            image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Lavant_St._Peter_und_Paul_Hochaltar_01.jpg/1024px-Lavant_St._Peter_und_Paul_Hochaltar_01.jpg",
        ),
    ]

    # Add the message with mixed content
    runner.append_message(
        {
            "role": "user",
            "content": mixed_content,
        },
    )

    # Example 3: Demonstrating file_id limitation
    print("\n=== Example 3: file_id limitation ===")
    print("Note: file_id is not supported for Completion API and will raise an error")

    try:
        file_content = [
            ResponseInputText(
                type="input_text",
                text="Can you describe this image file?",
            ),
            ResponseInputImage(
                type="input_image",
                detail="auto",
                file_id="file-12345",  # This will cause an error
            ),
        ]

        runner.append_message(
            {
                "role": "user",
                "content": file_content,
            },
        )

        # This will raise an error when trying to convert
        converted_messages = agent._convert_responses_to_completions_format(runner.messages)  # noqa: SLF001
        print("❌ This should not be reached!")

    except ValueError as e:
        print(f"✅ Expected error caught: {e}")
        # Clear the message that caused the error
        runner.messages.pop()

    # Print the internal message representation
    print("\n=== Internal message representation ===")
    for i, message in enumerate(runner.messages):
        print(f"Message {i + 1}:")
        if hasattr(message, "model_dump"):
            print(f"  {message.model_dump()}")
        else:
            print(f"  {message}")
        print()

    # Show how messages are converted for LLM API calls
    print("\n=== Converted messages for LLM API ===")
    # Note: In a real application, this conversion happens automatically
    # We're accessing the private method here just for demonstration
    converted_messages = agent._convert_responses_to_completions_format(runner.messages)  # noqa: SLF001
    for i, message in enumerate(converted_messages):
        print(f"Converted Message {i + 1}:")
        print(f"  {message}")
        print()

    print("\n=== Summary ===")
    print("The new Response API format allows you to:")
    print("1. Use ResponseInputText for text content")
    print("2. Use ResponseInputImage for image content with image_url")
    print("3. Mix text and images in a single message")
    print("4. Automatic conversion to legacy Completion API format for LLM calls")
    print("\n⚠️  Important limitations:")
    print("- file_id is NOT supported for Completion API - use image_url instead")
    print("- ResponseInputImage must have either file_id or image_url")
    print("\nThis provides better type safety and clearer separation between")
    print("internal representation and external API compatibility.")


if __name__ == "__main__":
    asyncio.run(main())
