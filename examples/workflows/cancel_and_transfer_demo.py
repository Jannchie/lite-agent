#!/usr/bin/env python3
"""
Demo showing two new features:
1. Cancellation of pending tool calls when user provides new input
2. function_call_output events for transfer_to_agent and transfer_to_parent
"""

import asyncio
import json
import logging

from funcall.decorators import tool
from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import AssistantTextContent, AssistantToolCall, AssistantToolCallResult, NewAssistantMessage, ToolCall, ToolCallFunction
from lite_agent.types.events import FunctionCallOutputEvent

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


@tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    await asyncio.sleep(2)  # Simulate long operation
    return f"The weather in {city} is sunny with a few clouds."


@tool
async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(2)  # Simulate long operation
    return f"The temperature in {city} is 25°C."


@tool
async def book_flight(destination: str, date: str) -> str:
    """Book a flight to a destination."""
    await asyncio.sleep(3)  # Simulate very long operation
    return f"Flight booked to {destination} on {date}"


async def demo_1_cancel_pending_tools():
    """Demo 1: Cancellation of pending tool calls"""
    print("=== Demo 1: Cancellation of Pending Tool Calls ===\n")

    agent = Agent(
        model="gpt-4.1-nano",
        name="WeatherBot",
        instructions="You are a helpful weather assistant.",
        tools=[get_weather, get_temperature, book_flight],
    )

    runner = Runner(agent)

    # Simulate a scenario where the agent made tool calls but user interrupts
    print("1. Simulating assistant making tool calls...")
    assistant_message = NewAssistantMessage(
        content=[
            AssistantTextContent(text="I'll check the weather and temperature for you, and also help you book that flight."),
            AssistantToolCall(
                call_id="weather_call",
                name="get_weather",
                arguments='{"city": "Tokyo"}',
            ),
            AssistantToolCall(
                call_id="temp_call",
                name="get_temperature",
                arguments='{"city": "Tokyo"}',
            ),
            AssistantToolCall(
                call_id="flight_call",
                name="book_flight",
                arguments='{"destination": "Paris", "date": "2024-12-25"}',
            ),
        ],
    )

    runner.messages.append(assistant_message)

    # Check pending calls
    pending_calls = runner._find_pending_tool_calls()
    print(f"Pending tool calls: {len(pending_calls)}")
    for call in pending_calls:
        print(f"  - {call.name} (call_id: {call.call_id})")

    print(f"Total messages before cancellation: {len(runner.messages)}")
    print(f"Content items in assistant message: {len(assistant_message.content)}")

    # User provides new input - this should cancel pending calls and generate events
    print("\n2. User provides new input: 'Actually, never mind about the weather. What's the capital of Japan?'")

    print("\n2a. Testing via run() method (integrated flow with event yielding):")
    # Test the integrated flow - collect cancellation events from run()
    # This will automatically call _cancel_pending_tool_calls() and yield events
    chunks = []
    try:
        async for chunk in runner.run("What's the capital of Japan?", includes=["function_call_output"]):
            chunks.append(chunk)
            if len(chunks) >= 3:  # Stop after getting cancellation events
                break
    except Exception as e:
        # Expected - no real LLM configured
        logger.debug("Expected exception in demo: %s", e)

    print(f"Events from run() method: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        if chunk.type == "function_call_output":
            print(f"  - Chunk {i + 1}: {chunk.type} for {chunk.tool_call_id}")
            print(f"    Tool: {chunk.name}")
            print(f"    Content: {chunk.content}")

    # Now demonstrate the direct method on fresh data
    print("\n2b. Testing direct cancellation method (returns events):")
    # Create a new runner with fresh pending calls
    runner2 = Runner(agent)
    runner2.messages.append(
        NewAssistantMessage(
            content=[
                AssistantTextContent(text="Let me help with a different task."),
                AssistantToolCall(
                    call_id="new_call_1",
                    name="get_weather",
                    arguments='{"city": "London"}',
                ),
                AssistantToolCall(
                    call_id="new_call_2",
                    name="get_temperature",
                    arguments='{"city": "London"}',
                ),
            ],
        ),
    )

    cancellation_events = runner2._cancel_pending_tool_calls()
    print(f"Cancellation events generated: {len(cancellation_events)}")
    for event in cancellation_events:
        print(f"  - Event: {event.type} for {event.tool_call_id} ({event.name})")
        print(f"    Content: {event.content}")
        print(f"    Execution time: {event.execution_time_ms}ms")

    print("\n3. After cancellation:")
    pending_calls_after = runner._find_pending_tool_calls()
    print(f"Pending tool calls: {len(pending_calls_after)}")

    print(f"Total messages after cancellation: {len(runner.messages)}")
    if runner.messages:
        assistant_msg = runner.messages[0]
        print(f"Content items in assistant message: {len(assistant_msg.content)}")

        # Show the cancellation results
        cancellation_results = [item for item in assistant_msg.content if isinstance(item, AssistantToolCallResult)]
        print(f"Cancellation results added: {len(cancellation_results)}")
        for result in cancellation_results:
            print(f"  - {result.call_id}: {result.output}")


async def demo_2_transfer_events():
    """Demo 2: function_call_output events for agent transfers"""
    print("\n\n=== Demo 2: Transfer Events ===\n")

    # Create agents with handoff relationships
    weather_agent = Agent(
        model="gpt-4",
        name="WeatherAgent",
        instructions="You specialize in weather information.",
        tools=[get_weather, get_temperature],
    )

    travel_agent = Agent(
        model="gpt-4",
        name="TravelAgent",
        instructions="You specialize in travel bookings.",
        tools=[book_flight],
    )

    main_agent = Agent(
        model="gpt-4",
        name="MainAgent",
        instructions="You are the main agent that coordinates with specialists.",
        handoffs=[weather_agent, travel_agent],
    )

    runner = Runner(main_agent)

    print("1. Testing transfer_to_agent with function_call_output events:")
    print(f"Current agent: {runner.agent.name}")

    # Simulate transfer_to_agent call

    transfer_call = ToolCall(
        type="function",
        id="transfer_001",
        function=ToolCallFunction(
            name="transfer_to_agent",
            arguments=json.dumps({"name": "WeatherAgent"}),
        ),
        index=0,
    )

    # Handle transfer and collect events
    chunks = [chunk async for chunk in runner._handle_tool_calls([transfer_call], ["function_call_output"])]

    print(f"Events generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, FunctionCallOutputEvent):
            print(f"  Event {i + 1}: {chunk.type} - {chunk.content}")
        else:
            print(f"  Event {i + 1}: {chunk.type}")

    print(f"Agent after transfer: {runner.agent.name}")

    print("\n2. Testing transfer_to_parent with function_call_output events:")

    parent_transfer_call = ToolCall(
        type="function",
        id="transfer_002",
        function=ToolCallFunction(
            name="transfer_to_parent",
            arguments="{}",
        ),
        index=0,
    )

    # Handle parent transfer and collect events
    chunks = [chunk async for chunk in runner._handle_tool_calls([parent_transfer_call], ["function_call_output"])]

    print(f"Events generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, FunctionCallOutputEvent):
            print(f"  Event {i + 1}: {chunk.type} - {chunk.content}")
        else:
            print(f"  Event {i + 1}: {chunk.type}")

    print(f"Agent after parent transfer: {runner.agent.name}")

    print("\n3. Testing multiple transfers (only first executes):")

    # Multiple transfer calls
    transfer_call_1 = ToolCall(
        type="function",
        id="multi_001",
        function=ToolCallFunction(
            name="transfer_to_agent",
            arguments=json.dumps({"name": "TravelAgent"}),
        ),
        index=0,
    )

    transfer_call_2 = ToolCall(
        type="function",
        id="multi_002",
        function=ToolCallFunction(
            name="transfer_to_agent",
            arguments=json.dumps({"name": "WeatherAgent"}),
        ),
        index=1,
    )

    chunks = [chunk async for chunk in runner._handle_tool_calls([transfer_call_1, transfer_call_2], ["function_call_output"])]

    print(f"Events generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, FunctionCallOutputEvent):
            print(f"  Event {i + 1}: {chunk.content}")
        else:
            print(f"  Event {i + 1}: {chunk.type}")

    print(f"Final agent: {runner.agent.name}")


async def main():
    """Run both demos"""
    print("🚀 Demo: New LiteAgent Features\n")

    # Demo 1: Cancellation of pending tools
    await demo_1_cancel_pending_tools()

    # Demo 2: Transfer events
    await demo_2_transfer_events()

    print("\n✅ All demos completed!")
    print("\nKey takeaways:")
    print("1. Pending tool calls are automatically cancelled when user provides new input")
    print("2. Cancellation now generates function_call_output events (not just history records)")
    print("3. Transfer operations generate function_call_output events like regular tools")
    print("4. Multiple transfers in one call: only the first executes, others get 'already executed' message")
    print("5. Both cancellation and transfer events are yielded through the run() method")


if __name__ == "__main__":
    asyncio.run(main())
