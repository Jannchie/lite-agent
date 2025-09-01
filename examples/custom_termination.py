import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import display_messages
from lite_agent.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny, 25°C."


async def mark_task_complete(task: str) -> str:
    """Mark a task as complete. This will terminate the agent."""
    return f"Task '{task}' has been marked as complete!"


async def save_report(content: str) -> str:
    """Save a report to file. This will also terminate the agent."""
    return f"Report saved: {content[:50]}..."


# Example 1: Single custom termination tool
agent1 = Agent(
    model="gpt-4.1",
    name="Task Assistant",
    instructions="You are a task assistant. Use the weather tool if needed, then mark the task complete when done.",
    tools=[get_weather, mark_task_complete],
    completion_condition="call",
    termination_tools=[mark_task_complete],  # Only this tool will terminate
)

# Example 2: Multiple termination tools
agent2 = Agent(
    model="gpt-4.1",
    name="Report Assistant",
    instructions="You are a report assistant. Get weather data and either save a report or mark the task complete.",
    tools=[get_weather, mark_task_complete, save_report],
    completion_condition="call",
    termination_tools=[mark_task_complete, save_report],  # Either tool will terminate
)

# Example 3: Using string names for termination tools
agent3 = Agent(
    model="gpt-4.1",
    name="String-based Assistant",
    instructions="You are an assistant. Get weather data and mark task complete when done.",
    tools=[get_weather, mark_task_complete],
    completion_condition="call",
    termination_tools=["mark_task_complete"],  # Using string name
)


async def test_single_termination():
    print("\n=== Testing Single Custom Termination Tool ===")
    runner = Runner(agent1)
    resp = runner.run(
        "Check the weather in Tokyo and mark the task as complete",
        includes=["assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        if chunk.type == "assistant_message":
            print(f"Assistant: {chunk.message.content}")

    print("\nFinal messages:")
    display_messages(runner.messages)


async def test_multiple_termination():
    print("\n=== Testing Multiple Termination Tools ===")
    runner = Runner(agent2)
    resp = runner.run(
        "Check weather in London and save a weather report",
        includes=["assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        if chunk.type == "assistant_message":
            print(f"Assistant: {chunk.message.content}")

    print("\nFinal messages:")
    display_messages(runner.messages)


async def test_string_termination():
    print("\n=== Testing String-based Termination ===")
    runner = Runner(agent3)
    resp = runner.run(
        "What's the weather like in Paris? Mark complete when done.",
        includes=["assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        if chunk.type == "assistant_message":
            print(f"Assistant: {chunk.message.content}")

    print("\nFinal messages:")
    display_messages(runner.messages)


async def main():
    await test_single_termination()
    await test_multiple_termination()
    await test_string_termination()


if __name__ == "__main__":
    asyncio.run(main())
