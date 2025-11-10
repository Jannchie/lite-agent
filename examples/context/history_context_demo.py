"""Demo showing how to access history messages in tool functions."""

import asyncio
import logging

from funcall import Context
from pydantic import BaseModel
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.context import HistoryContext
from lite_agent.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


# User-defined context data
class UserContext(BaseModel):
    user_id: str
    city: str


# Tool that only needs history messages
async def count_messages(ctx: Context[HistoryContext[None]]) -> str:
    """Count the number of messages in the conversation history."""
    messages = ctx.value.history_messages
    return f"The conversation has {len(messages)} messages in total."


# Tool that needs both history and user data
async def analyze_conversation(ctx: Context[HistoryContext[UserContext]]) -> str:
    """Analyze conversation history with user context."""
    messages = ctx.value.history_messages
    user_data = ctx.value.data

    if user_data is None:
        return f"Found {len(messages)} messages, but no user data available."

    user_id = user_data.user_id
    city = user_data.city

    # Analyze message content
    user_message_count = sum(1 for msg in messages if hasattr(msg, "role") and getattr(msg, "role", None) == "user")

    return f"Analysis for user {user_id} from {city}:\n- Total messages: {len(messages)}\n- User messages: {user_message_count}\n- Assistant messages: {len(messages) - user_message_count}"


# Tool that provides conversation summary
async def summarize_recent(ctx: Context[HistoryContext]) -> str:
    """Summarize the most recent messages."""
    messages = ctx.value.history_messages

    if len(messages) == 0:
        return "No conversation history available."

    # Get the last few messages
    recent_messages = messages[-3:]
    summary = f"Recent activity ({len(recent_messages)} messages):\n"

    for i, msg in enumerate(recent_messages, 1):
        # Handle different message types
        if hasattr(msg, "content"):
            content = getattr(msg, "content", "")
            if isinstance(content, list) and content:
                # Extract text from content list
                text_parts = []
                for item in content:
                    if hasattr(item, "text"):
                        text_parts.append(item.text)
                    elif isinstance(item, str):
                        text_parts.append(item)
                content_text = " ".join(text_parts)[:50] + "..."
            elif isinstance(content, str):
                content_text = content[:50] + "..."
            else:
                content_text = str(content)[:50] + "..."
        else:
            content_text = "No content"

        role = getattr(msg, "role", "unknown")
        summary += f"{i}. {role}: {content_text}\n"

    return summary


agent = Agent(
    model="gpt-4o-mini",
    name="History Assistant",
    instructions="You are an assistant that can analyze conversation history. Use the provided tools to help users understand their conversation patterns.",
    tools=[count_messages, analyze_conversation, summarize_recent],
)


async def demo_without_user_context():
    """Demo using tools without providing user context."""
    print("\n=== Demo 1: No user context (history only) ===")

    runner = Runner(agent)

    # Add some initial conversation
    runner.add_user_message("Hello, how are you?")
    runner.add_assistant_message("I'm doing well, thank you! How can I help you today?")
    runner.add_user_message("Can you count our messages?")

    resp = runner.run(
        "Please count how many messages we have exchanged so far.",
        includes=["function_call_output"],
    )

    async for chunk in resp:
        if chunk.type == "function_call_output":
            print(f"Tool result: {chunk.content}")


async def demo_with_user_context():
    """Demo using tools with user context data."""
    print("\n=== Demo 2: With user context ===")

    runner = Runner(agent)

    # Add some conversation history
    runner.add_user_message("Hi there!")
    runner.add_assistant_message("Hello! Nice to meet you.")
    runner.add_user_message("I'm from Beijing.")
    runner.add_assistant_message("That's great! Beijing is a wonderful city.")
    runner.add_user_message("Can you analyze our conversation?")

    # Provide user context
    user_ctx = UserContext(user_id="alice_123", city="Beijing")

    resp = runner.run(
        "Please analyze our conversation with my user information.",
        context=Context(user_ctx),
        includes=["function_call_output"],
    )

    async for chunk in resp:
        if chunk.type == "function_call_output":
            print(f"Tool result: {chunk.content}")


async def demo_conversation_summary():
    """Demo conversation summarization."""
    print("\n=== Demo 3: Conversation summary ===")

    runner = Runner(agent)

    # Build up a longer conversation
    conversation = [
        ("user", "Hello, I need help with Python."),
        ("assistant", "I'd be happy to help with Python! What specifically do you need help with?"),
        ("user", "I'm trying to understand async/await."),
        ("assistant", "Async/await is used for asynchronous programming in Python. It allows you to write concurrent code."),
        ("user", "Can you give me an example?"),
        ("assistant", "Sure! Here's a simple example: async def my_function(): await some_task()"),
        ("user", "That's helpful, thanks!"),
    ]

    for role, content in conversation:
        if role == "user":
            runner.add_user_message(content)
        else:
            runner.add_assistant_message(content)

    resp = runner.run(
        "Can you summarize our recent conversation?",
        includes=["function_call_output"],
    )

    async for chunk in resp:
        if chunk.type == "function_call_output":
            print(f"Tool result:\n{chunk.content}")


async def main():
    """Run all demos."""
    await demo_without_user_context()
    await demo_with_user_context()
    await demo_conversation_summary()

    print("\n=== Demos completed! ===")
    print("Key takeaways:")
    print("1. Tools automatically receive history_messages in context")
    print("2. Use Context[HistoryContext[None]] for history-only tools")
    print("3. Use Context[HistoryContext[YourDataType]] for tools that need both")
    print("4. Full type safety and IDE completion support")


if __name__ == "__main__":
    asyncio.run(main())
