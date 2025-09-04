"""Structured Output Demo

This example demonstrates how to use structured output with LiteAgent
to enforce specific response formats using Pydantic models.
"""

import asyncio
import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from lite_agent import Agent, Runner
from lite_agent.types import AssistantTextContent

# Set API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key-here")


class WeatherInfo(BaseModel):
    """Weather information structure."""

    city: str = Field(description="Name of the city")
    temperature: float = Field(description="Temperature in Celsius")
    condition: Literal["sunny", "cloudy", "rainy", "snowy"] = Field(description="Weather condition")
    humidity: int = Field(description="Humidity percentage", ge=0, le=100)
    description: str = Field(description="Brief description of the weather")


class TaskAnalysis(BaseModel):
    """Task analysis structure."""

    task_name: str = Field(description="Name of the task")
    priority: Literal["low", "medium", "high", "critical"] = Field(description="Task priority level")
    estimated_hours: float = Field(description="Estimated hours to complete", ge=0)
    dependencies: list[str] = Field(description="List of dependent tasks", default_factory=list)
    tags: list[str] = Field(description="Task tags for categorization", default_factory=list)


async def weather_demo():
    """Demonstrate structured output for weather information."""
    print("🌤️ Weather Information Demo with Structured Output")
    print("=" * 60)

    # Create agent with structured output
    weather_agent = Agent(
        model="gpt-4o-mini",
        name="WeatherAgent",
        instructions="""You are a weather information assistant.
        Provide weather information in the exact format specified by the response schema.
        Make the information realistic and helpful.""",
        response_format=WeatherInfo,
    )

    runner = Runner(weather_agent, api="completion")

    user_query = "What's the weather like in Tokyo today?"
    print(f"User: {user_query}")
    print("\nAgent Response:")

    chunks = runner.run(user_query)

    async for chunk in chunks:
        if chunk.type == "assistant_message":
            # Parse the structured response
            try:
                # Extract text content from assistant message
                text_content = ""
                for content in chunk.message.content:
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    # Try to parse as JSON
                    try:
                        weather_data = json.loads(text_content)
                        weather_info = WeatherInfo(**weather_data)
                        print(f"🏙️  City: {weather_info.city}")
                        print(f"🌡️  Temperature: {weather_info.temperature}°C")
                        print(f"☁️  Condition: {weather_info.condition}")
                        print(f"💧 Humidity: {weather_info.humidity}%")
                        print(f"📝 Description: {weather_info.description}")
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Raw response: {text_content}")
                        print(f"Note: Response parsing failed: {e}")

            except Exception as e:
                print(f"Error processing response: {e}")


async def task_analysis_demo():
    """Demonstrate structured output for task analysis."""
    print("\n📋 Task Analysis Demo with Structured Output")
    print("=" * 60)

    # Create agent with different structured output
    task_agent = Agent(
        model="gpt-4o-mini",
        name="TaskAnalyzer",
        instructions="""You are a project management assistant that analyzes tasks.
        Break down the given task and provide detailed analysis in the specified format.
        Be realistic about time estimates and dependencies.""",
        response_format=TaskAnalysis,
    )

    runner = Runner(task_agent, api="completion")

    user_query = "Analyze this task: Implement user authentication system for a web application"
    print(f"User: {user_query}")
    print("\nAgent Response:")

    chunks = runner.run(user_query)

    async for chunk in chunks:
        if chunk.type == "assistant_message":
            try:
                # Extract text content from assistant message
                text_content = ""
                for content in chunk.message.content:
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    # Try to parse as JSON
                    try:
                        task_data = json.loads(text_content)
                        task_analysis = TaskAnalysis(**task_data)
                        print(f"📝 Task: {task_analysis.task_name}")
                        print(f"⚡ Priority: {task_analysis.priority}")
                        print(f"⏱️  Estimated Hours: {task_analysis.estimated_hours}")
                        print(f"🔗 Dependencies: {', '.join(task_analysis.dependencies) if task_analysis.dependencies else 'None'}")
                        print(f"🏷️  Tags: {', '.join(task_analysis.tags) if task_analysis.tags else 'None'}")
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Raw response: {text_content}")
                        print(f"Note: Response parsing failed: {e}")

            except Exception as e:
                print(f"Error processing response: {e}")


async def runtime_response_format_demo():
    """Demonstrate changing response format at runtime."""
    print("\n🔄 Runtime Response Format Demo")
    print("=" * 60)

    # Create agent without initial response format
    flexible_agent = Agent(
        model="gpt-4o-mini",
        name="FlexibleAgent",
        instructions="You are a helpful assistant that can adapt to different response formats.",
    )

    runner = Runner(flexible_agent, api="completion")

    # First query with weather format
    print("First query with WeatherInfo format:")
    weather_query = "Tell me about the weather in London"
    chunks = runner.run(weather_query, response_format=WeatherInfo)

    async for chunk in chunks:
        if chunk.type == "assistant_message":
            try:
                text_content = ""
                for content in chunk.message.content:
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    try:
                        weather_data = json.loads(text_content)
                        weather_info = WeatherInfo(**weather_data)
                        print(f"Weather in {weather_info.city}: {weather_info.condition}, {weather_info.temperature}°C")
                    except (json.JSONDecodeError, ValueError):
                        print(f"Raw response: {text_content}")
            except Exception as e:
                print(f"Error: {e}")

    print("\nSecond query with TaskAnalysis format:")
    task_query = "Analyze this task: Create a mobile app prototype"
    chunks = runner.run(task_query, response_format=TaskAnalysis)

    async for chunk in chunks:
        if chunk.type == "assistant_message":
            try:
                text_content = ""
                for content in chunk.message.content:
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    try:
                        task_data = json.loads(text_content)
                        task_analysis = TaskAnalysis(**task_data)
                        print(f"Task: {task_analysis.task_name} (Priority: {task_analysis.priority})")
                    except (json.JSONDecodeError, ValueError):
                        print(f"Raw response: {text_content}")
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Run all structured output demonstrations."""
    print("🚀 LiteAgent Structured Output Demonstrations")
    print("=" * 60)

    try:
        await weather_demo()
        await task_analysis_demo()
        await runtime_response_format_demo()

        print("\n✅ All demonstrations completed successfully!")
        print("\nKey features demonstrated:")
        print("• Agent initialization with response_format parameter")
        print("• Runtime response_format specification in run() method")
        print("• Automatic JSON schema generation from Pydantic models")
        print("• Type-safe structured responses")
        print("• Multiple response format support")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(main())
