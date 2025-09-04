"""Structured Output API Comparison

This example compares how structured output works with both
completion and responses APIs, showing their differences.
"""

import asyncio
import json
import os
from typing import Literal

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

    return text


async def test_api(api_name: Literal["completion", "responses"]) -> bool:
    """Test structured output with specific API."""
    print(f"\n{'=' * 20} {api_name.upper()} API {'=' * 20}")

    agent = Agent(
        model="gpt-4o-mini",
        name="PersonAgent",
        instructions="""You must extract person information and return it as JSON using EXACTLY these field names:
        - name: the person's full name
        - age: the person's age as a number
        - occupation: the person's job/profession
        - city: the city where the person lives
        Do not use any other field names. Use exactly: name, age, occupation, city.""",
        response_format=PersonInfo,
    )

    runner = Runner(agent, api=api_name)
    user_input = "Tell me about Sarah Chen, a 28-year-old doctor from Boston"

    print(f"Input: {user_input}")

    chunks = runner.run(user_input)

    async for chunk in chunks:
        if chunk.type == "assistant_message":
            try:
                text_content = ""
                for content in chunk.message.content:
                    if isinstance(content, AssistantTextContent):
                        text_content += content.text

                if text_content.strip():
                    print(f"Raw Response: {text_content}")

                    # Extract JSON (handles both plain JSON and markdown-wrapped)
                    json_text = extract_json_from_response(text_content)
                    if json_text != text_content:
                        print(f"Extracted JSON: {json_text}")

                    try:
                        person_data = json.loads(json_text)
                        person = PersonInfo(**person_data)

                        print("✅ Success!")
                        print(f"   Name: {person.name}")
                        print(f"   Age: {person.age}")
                        print(f"   Occupation: {person.occupation}")
                        print(f"   City: {person.city}")
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"❌ JSON parsing error: {e}")
                        return False
                    return True

            except Exception as e:
                print(f"❌ Error: {e}")
                return False

    return False


async def main():
    """Compare both APIs."""
    print("🔄 Structured Output API Comparison")
    print("=" * 60)

    # Test both APIs
    completion_success = await test_api("completion")
    responses_success = await test_api("responses")

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"Completion API: {'✅ SUCCESS' if completion_success else '❌ FAILED'}")
    print(f"Responses API:  {'✅ SUCCESS' if responses_success else '❌ FAILED'}")

    print("\n💡 Key Differences:")
    print("• Completion API: Returns pure JSON")
    print("• Responses API: May wrap JSON in ```json markdown blocks")
    print("• Both support structured output, just different formatting")
    print("• Completion API generally more reliable for strict JSON schemas")

    if completion_success and responses_success:
        print("\n🎉 Both APIs working perfectly with structured output!")
    elif completion_success:
        print("\n⚠️  Recommend using Completion API for structured output")
    else:
        print("\n❌ Issues detected - check your API key and model access")


if __name__ == "__main__":
    asyncio.run(main())
