"""Integration tests for structured output functionality."""

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm import CustomStreamWrapper
from pydantic import BaseModel, Field

from lite_agent import Agent, Runner
from lite_agent.client import LiteLLMClient
from lite_agent.types import (
    AssistantMessageEvent,
    AssistantMessageMeta,
    AssistantTextContent,
    NewAssistantMessage,
)


class PersonInfo(BaseModel):
    """Test person information model."""

    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age", ge=0, le=150)
    city: str = Field(description="Person's city")


class TestStructuredOutputIntegration:
    """Integration tests for structured output."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock LiteLLM client."""
        client = MagicMock(spec=LiteLLMClient)
        client.model = "gpt-4o-mini"
        return client

    @pytest.fixture
    def mock_completion_response(self):
        """Create mock completion response with structured output."""
        person_data = {
            "name": "John Doe",
            "age": 30,
            "city": "New York",
        }

        # Create mock assistant message
        message = NewAssistantMessage(
            content=[AssistantTextContent(text=json.dumps(person_data))],
            meta=AssistantMessageMeta(model="gpt-4o-mini"),
        )

        # Create assistant message event
        event = AssistantMessageEvent(message=message)

        # Create async generator that yields the event
        async def async_gen() -> AsyncGenerator[AssistantMessageEvent, None]:
            yield event

        # Wrap in CustomStreamWrapper to match expected type
        class MockCustomStreamWrapper(CustomStreamWrapper):
            def __init__(self, async_iterable):
                self._async_iterable = async_iterable

            def __aiter__(self):
                return self._async_iterable.__aiter__()

        return MockCustomStreamWrapper(async_gen())

    @pytest.fixture
    def agent_with_structured_output(self, mock_client):
        """Create agent with structured output format."""
        return Agent(
            model=mock_client,
            name="TestAgent",
            instructions="Extract person information",
            response_format=PersonInfo,
        )

    def test_agent_has_response_format(self, agent_with_structured_output):
        """Test that agent has the correct response format."""
        assert agent_with_structured_output.response_format == PersonInfo

    @pytest.mark.asyncio
    async def test_completion_with_response_format(self, agent_with_structured_output, mock_completion_response):
        """Test completion with response format."""
        # Mock the client's completion method
        agent_with_structured_output.client.completion = AsyncMock(return_value=mock_completion_response)

        # Call completion with messages
        messages = []
        chunks = []

        async for chunk in await agent_with_structured_output.completion(messages):
            chunks.append(chunk)

        # Verify completion was called with response_format
        agent_with_structured_output.client.completion.assert_called_once()
        call_args = agent_with_structured_output.client.completion.call_args

        # Check that response_format was passed
        assert "response_format" in call_args.kwargs
        # The response_format should be the PersonInfo model
        assert call_args.kwargs["response_format"] == PersonInfo

    @pytest.mark.asyncio
    async def test_responses_with_response_format(self, agent_with_structured_output, mock_completion_response):
        """Test responses with response format."""
        # Mock the client's responses method
        agent_with_structured_output.client.responses = AsyncMock(return_value=mock_completion_response)

        # Call responses with messages
        messages = []

        async for _chunk in await agent_with_structured_output.responses(messages):
            break  # Just test the call

        # Verify responses was called with response_format
        agent_with_structured_output.client.responses.assert_called_once()
        call_args = agent_with_structured_output.client.responses.call_args

        # Check that response_format was passed
        assert "response_format" in call_args.kwargs
        assert call_args.kwargs["response_format"] == PersonInfo

    @pytest.mark.asyncio
    async def test_runtime_response_format_override(self, mock_client, mock_completion_response):
        """Test overriding response format at runtime."""
        # Create agent without response format
        agent = Agent(
            model=mock_client,
            name="TestAgent",
            instructions="Test instructions",
        )

        # Mock the client methods
        agent.client.completion = AsyncMock(return_value=mock_completion_response)
        agent.client.responses = AsyncMock(return_value=mock_completion_response)

        # Test completion with runtime response format
        messages = []

        # This should use the runtime response_format, not the agent's default
        async for _chunk in await agent.completion(messages, response_format=PersonInfo):
            break

        # Verify the runtime response_format was used
        call_args = agent.client.completion.call_args
        assert call_args.kwargs["response_format"] == PersonInfo

    def test_runner_with_structured_output(self, agent_with_structured_output):
        """Test Runner with structured output agent."""
        runner = Runner(agent_with_structured_output)

        # The runner should work with structured output agents
        assert runner.agent.response_format == PersonInfo

    @pytest.mark.asyncio
    async def test_response_format_precedence(self, mock_client, mock_completion_response):
        """Test that method parameter takes precedence over agent setting."""
        # Create agent with default response format
        agent = Agent(
            model=mock_client,
            name="TestAgent",
            instructions="Test instructions",
            response_format=PersonInfo,
        )

        # Mock client method
        agent.client.completion = AsyncMock(return_value=mock_completion_response)

        # Define alternative format
        class AlternativeFormat(BaseModel):
            message: str

        messages = []

        # Call with different response_format - this should override the agent's default
        async for _chunk in await agent.completion(messages, response_format=AlternativeFormat):
            break

        # Verify the method parameter was used, not the agent's default
        call_args = agent.client.completion.call_args
        assert call_args.kwargs["response_format"] == AlternativeFormat
        assert call_args.kwargs["response_format"] != PersonInfo
