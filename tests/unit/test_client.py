"""Tests for client module."""

import os
from unittest.mock import Mock, patch

import pytest

from lite_agent.client import (
    LiteLLMClient,
    LLMConfig,
    parse_reasoning_config,
)


def test_llm_config_creation():
    """Test LLMConfig creation with various parameters."""
    config = LLMConfig(
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        stop=["END"],
    )

    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.top_p == 0.9
    assert config.frequency_penalty == 0.1
    assert config.presence_penalty == 0.2
    assert config.stop == ["END"]


def test_llm_config_defaults():
    """Test LLMConfig creation with default values."""
    config = LLMConfig()

    assert config.temperature is None
    assert config.max_tokens is None
    assert config.top_p is None
    assert config.frequency_penalty is None
    assert config.presence_penalty is None
    assert config.stop is None


def test_parse_reasoning_config_none():
    """Test parsing None reasoning config."""
    effort, config = parse_reasoning_config(None)
    assert effort is None
    assert config is None


def test_parse_reasoning_config_string_valid():
    """Test parsing valid string reasoning config."""
    for value in ["minimal", "low", "medium", "high"]:
        effort, config = parse_reasoning_config(value)
        assert effort == value
        assert config is None


def test_parse_reasoning_config_string_invalid():
    """Test parsing invalid string reasoning config."""
    effort, config = parse_reasoning_config("invalid_string")
    assert effort is None
    assert config is None


def test_parse_reasoning_config_dict():
    """Test parsing dict reasoning config."""
    test_dict = {"type": "enabled", "budget_tokens": 2048}
    effort, config = parse_reasoning_config(test_dict)
    assert effort is None
    assert config == test_dict


def test_parse_reasoning_config_bool_true():
    """Test parsing True boolean reasoning config."""
    effort, config = parse_reasoning_config(True)
    assert effort == "medium"
    assert config is None


def test_parse_reasoning_config_bool_false():
    """Test parsing False boolean reasoning config."""
    effort, config = parse_reasoning_config(False)
    assert effort is None
    assert config is None


def test_parse_reasoning_config_invalid_type():
    """Test parsing invalid type reasoning config."""
    effort, config = parse_reasoning_config(123)  # Invalid type
    assert effort is None
    assert config is None


def test_litellm_client_init():
    """Test LiteLLMClient initialization."""
    client = LiteLLMClient(
        model="gpt-4",
        api_key="test-key",
        api_base="https://api.test.com",
        api_version="2023-12-01",
        reasoning="medium",
        temperature=0.8,
        max_tokens=500,
    )

    assert client.model == "gpt-4"
    assert client.api_key == "test-key"
    assert client.api_base == "https://api.test.com"
    assert client.api_version == "2023-12-01"
    assert client.reasoning_effort == "medium"
    assert client.thinking_config is None
    assert client.llm_config.temperature == 0.8
    assert client.llm_config.max_tokens == 500


def test_litellm_client_init_with_llm_config():
    """Test LiteLLMClient initialization with explicit LLMConfig."""
    llm_config = LLMConfig(temperature=0.5, max_tokens=800)
    client = LiteLLMClient(
        model="gpt-4",
        llm_config=llm_config,
    )

    assert client.llm_config.temperature == 0.5
    assert client.llm_config.max_tokens == 800


def test_litellm_client_init_reasoning_dict():
    """Test LiteLLMClient initialization with dict reasoning config."""
    reasoning_config = {"type": "enabled", "budget_tokens": 1000}
    client = LiteLLMClient(
        model="gpt-4",
        reasoning=reasoning_config,
    )

    assert client.reasoning_effort is None
    assert client.thinking_config == reasoning_config


def test_litellm_client_resolve_reasoning_params_override():
    """Test _resolve_reasoning_params with override."""
    client = LiteLLMClient(
        model="gpt-4",
        reasoning="low",  # instance default
    )

    # Override with different reasoning
    effort, config = client._resolve_reasoning_params("high")
    assert effort == "high"
    assert config is None

    # Override with None should use instance default
    effort, config = client._resolve_reasoning_params(None)
    assert effort == "low"
    assert config is None


def test_litellm_client_resolve_reasoning_params_dict():
    """Test _resolve_reasoning_params with dict reasoning."""
    test_dict = {"type": "enabled", "budget_tokens": 512}
    client = LiteLLMClient(
        model="gpt-4",
        reasoning=test_dict,
    )

    # Use instance default
    effort, config = client._resolve_reasoning_params(None)
    assert effort is None
    assert config == test_dict


@pytest.mark.asyncio
async def test_litellm_client_completion():
    """Test LiteLLMClient completion method."""
    client = LiteLLMClient(
        model="gpt-4",
        api_key="test-key",
        reasoning="medium",
        temperature=0.7,
        max_tokens=1000,
    )

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test"}}]

    with patch("lite_agent.client.litellm.acompletion") as mock_completion:
        mock_completion.return_value = Mock()

        await client.completion(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            streaming=True,
        )

        # Verify the call
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]

        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["messages"] == messages
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["stream"] is True
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_litellm_client_completion_reasoning_override():
    """Test LiteLLMClient completion with reasoning override."""
    client = LiteLLMClient(
        model="gpt-4",
        reasoning="low",  # Default
    )

    messages = [{"role": "user", "content": "Hello"}]

    with patch("lite_agent.client.litellm.acompletion") as mock_completion:
        mock_completion.return_value = Mock()

        await client.completion(
            messages=messages,
            reasoning="high",  # Override
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_litellm_client_responses():
    """Test LiteLLMClient responses method."""
    client = LiteLLMClient(
        model="gpt-4",
        api_key="test-key",
        api_base="https://api.test.com",
        reasoning={"type": "enabled"},
        temperature=0.9,
    )

    messages = [{"role": "user", "input": [{"type": "input_text", "text": "Hello"}]}]
    tools = [{"type": "function", "name": "test"}]

    with patch("lite_agent.client.litellm.aresponses") as mock_responses:
        mock_responses.return_value = Mock()

        await client.responses(
            messages=messages,
            tools=tools,
            tool_choice="required",
            streaming=False,
        )

        # Verify environment variable is set
        assert os.environ["DISABLE_AIOHTTP_TRANSPORT"] == "True"

        # Verify the call
        mock_responses.assert_called_once()
        call_kwargs = mock_responses.call_args[1]

        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["input"] == messages
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "required"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["api_base"] == "https://api.test.com"
        assert call_kwargs["stream"] is False
        assert call_kwargs["store"] is False
        assert call_kwargs["thinking"] == {"type": "enabled"}
        assert call_kwargs["temperature"] == 0.9


@pytest.mark.asyncio
async def test_litellm_client_completion_minimal_config():
    """Test LiteLLMClient completion with minimal configuration."""
    client = LiteLLMClient(model="gpt-3.5-turbo")

    messages = [{"role": "user", "content": "Test"}]

    with patch("lite_agent.client.litellm.acompletion") as mock_completion:
        mock_completion.return_value = Mock()

        await client.completion(messages=messages)

        call_kwargs = mock_completion.call_args[1]

        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert call_kwargs["messages"] == messages
        assert call_kwargs["tools"] is None
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["stream"] is True
        # These shouldn't be present when not configured
        assert "reasoning_effort" not in call_kwargs
        assert "thinking" not in call_kwargs
        assert "temperature" not in call_kwargs


@pytest.mark.asyncio
async def test_litellm_client_responses_minimal_config():
    """Test LiteLLMClient responses with minimal configuration."""
    client = LiteLLMClient(model="gpt-4")

    messages = [{"role": "user", "input": []}]

    with patch("lite_agent.client.litellm.aresponses") as mock_responses:
        mock_responses.return_value = Mock()

        await client.responses(messages=messages)

        call_kwargs = mock_responses.call_args[1]

        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["input"] == messages
        assert call_kwargs["tools"] is None
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["stream"] is True
        assert call_kwargs["store"] is False
        # These shouldn't be present when not configured
        assert "reasoning_effort" not in call_kwargs
        assert "thinking" not in call_kwargs
