import asyncio
import logging

from funcall.decorators import tool
from rich.logging import RichHandler

from lite_agent.agent import Agent
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
    await asyncio.sleep(1)
    return f"The weather in {city} is sunny with a few clouds."


async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    await asyncio.sleep(1)
    return f"The temperature in {city} is 25°C."


@tool(require_confirmation=True)
async def send_email(to: str, subject: str) -> str:
    """Send an email (decorated with require_confirmation=True)."""
    await asyncio.sleep(1)
    return f"Email sent to {to} with subject: {subject}"


async def main():
    # Create agent with stop_before_functions parameter (using callable)
    agent = Agent(
        model="gpt-4.1-nano",
        name="Weather Assistant",
        instructions="You are a helpful weather assistant. Provide friendly and informative responses.",
        tools=[get_weather, get_temperature, send_email],
        stop_before_tools=[get_temperature],  # Stop before calling get_temperature (using callable)
    )

    runner = Runner(agent)

    print("=== Test 1: stop_before_functions in constructor ===")
    resp = runner.run(
        "What is the weather in New York? And what is the temperature there?",
        includes=["usage", "assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        logger.info(chunk)

    print(f"\nHas require confirm tools: {await runner.has_require_confirm_tools()}")

    # Continue execution
    resp = runner.run_continue_stream(
        includes=["usage", "assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        logger.info(chunk)

    print("\n=== Test 2: Dynamic function addition (using callable) ===")
    # Add another function dynamically using callable
    agent.add_stop_before_function(get_weather)
    print(f"Stop before functions: {agent.get_stop_before_functions()}")

    runner2 = Runner(agent)
    resp = runner2.run(
        "What is the weather in Tokyo?",
        includes=["usage", "assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        logger.info(chunk)

    print(f"\nHas require confirm tools: {await runner2.has_require_confirm_tools()}")

    # Continue execution
    resp = runner2.run_continue_stream(
        includes=["usage", "assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        logger.info(chunk)

    print("\n=== Test 3: Decorator-based require_confirmation still works ===")
    agent.clear_stop_before_functions()
    print(f"Stop before functions cleared: {agent.get_stop_before_functions()}")

    runner3 = Runner(agent)
    resp = runner3.run(
        "Send an email to john@example.com with subject 'Weather Update'",
        includes=["usage", "assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        logger.info(chunk)

    print(f"\nHas require confirm tools (decorator): {await runner3.has_require_confirm_tools()}")

    # Continue execution
    resp = runner3.run_continue_stream(
        includes=["usage", "assistant_message", "function_call", "function_call_output"],
    )
    async for chunk in resp:
        logger.info(chunk)


if __name__ == "__main__":
    asyncio.run(main())
