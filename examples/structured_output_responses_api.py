"""Structured Output with Responses API Example

Shows how to handle structured output with the responses API,
which may return JSON wrapped in markdown code blocks.
"""

import asyncio
import json
import os
import re

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


def extract_json_from_response(text: str) -> str:
    """Extract JSON from potentially markdown-wrapped response."""
    text = text.strip()

    # Remove markdown code block wrapper if present
    if text.startswith("```json") and text.endswith("```"):
        return text[7:-3].strip()
    if text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()

    # Try to find JSON object using regex
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return text


async def test_responses_api():
    """Test structured output with responses API."""
    print("Structured Output with Responses API")
    print("=" * 45)

    # Create agent with structured output
    agent = Agent(
        model="gpt-4o-mini",
        name="PersonAgent",
        instructions="Extract person information and return it as JSON in the exact format specified by the schema. Be precise with field names.",
        response_format=PersonInfo,
    )

    runner = Runner(agent, api="responses")  # Using responses API

    # Test with person information
    user_input = "Tell me about John Smith, a 30-year-old software engineer living in San Francisco"
    print(f"Input: {user_input}")
    print("API: responses")
    print("\nStructured Output:")

    chunks = runner.run(user_input)

    async for chunk in chunks:
        if chunk.type == "assistant_message":
            try:
                # Extract text content
                text_content = ""
                for content in chunk.message.content:
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    print(f"Raw response: {text_content}")

                    # Extract JSON from potentially wrapped response
                    json_text = extract_json_from_response(text_content)
                    print(f"Extracted JSON: {json_text}")

                    try:
                        person_data = json.loads(json_text)
                        person = PersonInfo(**person_data)

                        print("\n✅ Successfully parsed structured output:")
                        print(f"Name: {person.name}")
                        print(f"Age: {person.age}")
                        print(f"Occupation: {person.occupation}")
                        print(f"City: {person.city}")

                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"❌ JSON parsing error: {e}")

            except Exception as e:
                print(f"❌ Error: {e}")


async def compare_apis():
    """Compare completion vs responses API output."""
    print("\n" + "=" * 50)
    print("API Comparison")
    print("=" * 50)

    user_input = "Create info for Alice Johnson, age 28, teacher from Boston"

    for api_name in ["completion", "responses"]:
        print(f"\n--- {api_name.upper()} API ---")

        agent = Agent(
            model="gpt-4o-mini",
            name="PersonAgent",
            instructions="Return person information as JSON with exact field names: name, age, occupation, city",
            response_format=PersonInfo,
        )

        runner = Runner(agent, api=api_name)
        chunks = runner.run(user_input)

        async for chunk in chunks:
            if chunk.type == "assistant_message":
                text_content = ""
                for content in chunk.message.content:
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    print(f"Raw output: {text_content}")

                    # Try to parse
                    try:
                        json_text = extract_json_from_response(text_content)
                        data = json.loads(json_text)
                        person = PersonInfo(**data)
                        print(f"✅ Parsed: {person.name}, {person.age}, {person.occupation}, {person.city}")
                    except Exception as e:
                        print(f"❌ Parse failed: {e}")
                break


async def main():
    """Run both tests."""
    try:
        await test_responses_api()
        await compare_apis()

        print("\n" + "=" * 50)
        print("Summary:")
        print("• Responses API may wrap JSON in markdown code blocks")
        print("• Use extract_json_from_response() to handle both formats")
        print("• Both APIs support structured output, just different formatting")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(main())
