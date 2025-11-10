"""Tests for OpenAI client module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from lite_agent.client import LLMConfig, OpenAIClient, parse_reasoning_config


def _build_openai_mock() -> Mock:
    """Create a mock AsyncOpenAI client with async methods."""

    mock_client = Mock()

    chat_interface = Mock()
    chat_completions = Mock()
    chat_completions.create = AsyncMock(return_value=Mock())
    chat_interface.completions = chat_completions
    mock_client.chat = chat_interface

    responses_interface = Mock()
    responses_interface.create = AsyncMock(return_value=Mock())
    mock_client.responses = responses_interface

    return mock_client


def test_llm_config_creation() -> None:
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


def test_llm_config_defaults() -> None:
    """Test LLMConfig creation with default values."""

    config = LLMConfig()

    assert config.temperature is None
    assert config.max_tokens is None
    assert config.top_p is None
    assert config.frequency_penalty is None
    assert config.presence_penalty is None
    assert config.stop is None


def test_parse_reasoning_config_variants() -> None:
    """Verify parse_reasoning_config handles supported input forms."""

    effort, config = parse_reasoning_config(None)
    assert effort is None
    assert config is None

    for value in ["minimal", "low", "medium", "high"]:
        effort, config = parse_reasoning_config(value)
        assert effort == value
        assert config is None

    bool_effort, bool_config = parse_reasoning_config(True)
    assert bool_effort == "medium"
    assert bool_config is None

    bool_effort, bool_config = parse_reasoning_config(False)
    assert bool_effort is None
    assert bool_config is None

    test_dict = {"type": "enabled", "budget_tokens": 2048}
    dict_effort, dict_config = parse_reasoning_config(test_dict)
    assert dict_effort is None
    assert dict_config == test_dict

    invalid_effort, invalid_config = parse_reasoning_config(123)  # type: ignore[arg-type]
    assert invalid_effort is None
    assert invalid_config is None


def test_openai_client_init() -> None:
    """OpenAIClient should configure AsyncOpenAI with provided credentials."""

    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()

        client = OpenAIClient(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.test.com",
            reasoning="medium",
            temperature=0.8,
            max_tokens=500,
        )

    mock_async_openai.assert_called_once_with(api_key="test-key", base_url="https://api.test.com")
    assert client.model == "gpt-4"
    assert client.reasoning_effort == "medium"
    assert client.thinking_config is None
    assert client.llm_config.temperature == 0.8
    assert client.llm_config.max_tokens == 500


def test_openai_client_init_with_llm_config() -> None:
    """LLMConfig can be supplied directly."""

    llm_config = LLMConfig(temperature=0.5, max_tokens=800)
    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()
        client = OpenAIClient(model="gpt-4", llm_config=llm_config)

    assert client.llm_config.temperature == 0.5
    assert client.llm_config.max_tokens == 800


def test_openai_client_init_reasoning_dict() -> None:
    """Reasoning dict should populate thinking_config."""

    reasoning_config = {"type": "enabled", "budget_tokens": 1000}
    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()
        client = OpenAIClient(model="gpt-4", reasoning=reasoning_config)

    assert client.reasoning_effort is None
    assert client.thinking_config == reasoning_config


def test_openai_client_resolve_reasoning_params_override() -> None:
    """Method should respect overrides and fall back to defaults."""

    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()
        client = OpenAIClient(model="gpt-4", reasoning="low")

    effort, config = client._resolve_reasoning_params("high")
    assert effort == "high"
    assert config is None

    effort, config = client._resolve_reasoning_params(None)
    assert effort == "low"
    assert config is None


def test_openai_client_resolve_reasoning_params_dict() -> None:
    """Dict reasoning should be returned when no override is provided."""

    reasoning_config = {"type": "enabled", "budget_tokens": 512}
    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()
        client = OpenAIClient(model="gpt-4", reasoning=reasoning_config)

    effort, config = client._resolve_reasoning_params(None)
    assert effort is None
    assert config == reasoning_config


@pytest.mark.asyncio
async def test_openai_client_completion() -> None:
    """completion should forward parameters to the OpenAI SDK."""

    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        openai_mock = _build_openai_mock()
        mock_async_openai.return_value = openai_mock
        client = OpenAIClient(
            model="gpt-4",
            reasoning="medium",
            temperature=0.7,
            max_tokens=1000,
        )

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test"}}]

    await client.completion(messages=messages, tools=tools, tool_choice="auto", streaming=True)

    mock_async_openai.return_value.chat.completions.create.assert_awaited_once()
    call_kwargs = mock_async_openai.return_value.chat.completions.create.await_args.kwargs

    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["messages"] == messages
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["stream"] is True
    assert call_kwargs["reasoning_effort"] == "medium"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_openai_client_completion_reasoning_override() -> None:
    """Method-level reasoning override should win."""

    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()
        client = OpenAIClient(model="gpt-4", reasoning="low")

    await client.completion(messages=[{"role": "user", "content": "Hi"}], reasoning="high")

    call_kwargs = mock_async_openai.return_value.chat.completions.create.await_args.kwargs
    assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_openai_client_responses() -> None:
    """responses should forward configuration to the Responses API."""

    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        openai_mock = _build_openai_mock()
        mock_async_openai.return_value = openai_mock
        client = OpenAIClient(
            model="gpt-4",
            reasoning={"type": "enabled"},
            temperature=0.9,
        )

    messages = [{"role": "user", "input": [{"type": "input_text", "text": "Hello"}]}]
    tools = [{"type": "function", "name": "test"}]

    await client.responses(messages=messages, tools=tools, tool_choice="required", streaming=False)

    mock_async_openai.return_value.responses.create.assert_awaited_once()
    call_kwargs = mock_async_openai.return_value.responses.create.await_args.kwargs

    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["input"] == messages
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "required"
    assert "stream" not in call_kwargs
    assert call_kwargs["store"] is False
    assert "reasoning" not in call_kwargs
    assert call_kwargs["temperature"] == 0.9


@pytest.mark.asyncio
async def test_openai_client_completion_minimal_config() -> None:
    """Minimal configuration should not send optional parameters."""

    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()
        client = OpenAIClient(model="gpt-3.5-turbo")

    await client.completion(messages=[{"role": "user", "content": "Test"}])

    call_kwargs = mock_async_openai.return_value.chat.completions.create.await_args.kwargs

    assert call_kwargs["model"] == "gpt-3.5-turbo"
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["stream"] is True
    assert "tools" not in call_kwargs
    assert "reasoning_effort" not in call_kwargs
    assert "temperature" not in call_kwargs


@pytest.mark.asyncio
async def test_openai_client_responses_minimal_config() -> None:
    """Minimal configuration for responses should only set required fields."""

    with patch("lite_agent.client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = _build_openai_mock()
        client = OpenAIClient(model="gpt-4")

    await client.responses(messages=[{"role": "user", "input": []}])

    call_kwargs = mock_async_openai.return_value.responses.create.await_args.kwargs

    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["input"] == [{"role": "user", "input": []}]
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["stream"] is True
    assert call_kwargs["store"] is False
    assert "tools" not in call_kwargs
    assert "reasoning" not in call_kwargs
    assert "temperature" not in call_kwargs
