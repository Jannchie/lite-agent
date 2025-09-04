"""Basic Structured Output Example

Simple example showing how to use Pydantic models for structured output.
Works with both completion and responses APIs.
"""

import asyncio
import json
import os

from pydantic import BaseModel, Field

from lite_agent import Agent, Runner
from lite_agent.types import AssistantTextContent

# Set API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key-here")


class PersonInfo(BaseModel):
    """Person information structure."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    occupation: str = Field(description="Person's job or profession")
    city: str = Field(description="City where the person lives")


async def main():
    """Basic structured output example."""
    print("Basic Structured Output Example")
    print("=" * 40)

    # Create agent with structured output
    agent = Agent(
        model="gpt-4o-mini",
        name="PersonAgent",
        instructions="Extract person information and return it as JSON in the exact format specified by the schema. Be precise with field names.",
        response_format=PersonInfo,
    )

    runner = Runner(agent)  # Completion API has more reliable structured output

    # Test with person information
    user_input = "Tell me about John Smith, a 30-year-old software engineer living in San Francisco"
    print(f"Input: {user_input}")
    print("\nStructured Output:")

    chunks = runner.run(user_input)

    async for chunk in chunks:
        if chunk.type == "assistant_message":
            try:
                # Extract text content
                text_content = ""
                for content in chunk.message.content:
                    # 只处理文本内容类型
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    # Handle markdown-wrapped JSON (responses API may wrap in code blocks)
                    json_text = text_content.strip()
                    if json_text.startswith("```json") and json_text.endswith("```"):
                        json_text = json_text[7:-3].strip()
                    elif json_text.startswith("```") and json_text.endswith("```"):
                        json_text = json_text[3:-3].strip()

                    # Parse the JSON response
                    try:
                        person_data = json.loads(json_text)
                        person = PersonInfo(**person_data)

                        print(f"Name: {person.name}")
                        print(f"Age: {person.age}")
                        print(f"Occupation: {person.occupation}")
                        print(f"City: {person.city}")

                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Raw response: {text_content}")
                        print(f"Parsing error: {e}")

            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
