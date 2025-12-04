"""Tests for response handlers."""

from collections.abc import AsyncGenerator
from unittest.mock import Mock, patch

import pytest

from lite_agent.response_handlers.completion import CompletionResponseHandler
from lite_agent.response_handlers.responses import ResponsesAPIHandler
from lite_agent.types.events import AssistantMessageEvent, UsageEvent
from lite_agent.types.messages import AssistantTextContent, AssistantToolCall


@pytest.mark.asyncio
async def test_completion_handler_non_streaming_with_text():
    """Test completion handler with non-streaming text response."""
    handler = CompletionResponseHandler()

    # Mock response
    mock_response = Mock()
    mock_response.model = "gpt-4"
    mock_choice = Mock()
    mock_choice.message = Mock()
    mock_choice.message.content = "Hello, world!"
    mock_choice.message.tool_calls = None
    mock_response.choices = [mock_choice]

    # Mock usage
    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_response.usage = mock_usage

    # Collect chunks
    chunks = []
    async for chunk in handler._handle_non_streaming(mock_response):
        chunks.append(chunk)

    # Verify results
    assert len(chunks) == 2  # message + usage

    # Check message chunk
    message_chunk = chunks[0]
    assert isinstance(message_chunk, AssistantMessageEvent)
    assert len(message_chunk.message.content) == 1
    assert isinstance(message_chunk.message.content[0], AssistantTextContent)
    assert message_chunk.message.content[0].text == "Hello, world!"
    assert message_chunk.message.meta.model == "gpt-4"

    # Check usage chunk
    usage_chunk = chunks[1]
    assert isinstance(usage_chunk, UsageEvent)
    assert usage_chunk.usage.input_tokens == 10
    assert usage_chunk.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_completion_handler_non_streaming_with_tool_calls():
    """Test completion handler with tool calls."""
    handler = CompletionResponseHandler()

    # Mock response with tool calls
    mock_response = Mock()
    mock_response.model = "gpt-4"
    mock_choice = Mock()
    mock_choice.message = Mock()
    mock_choice.message.content = None

    # Mock tool call
    mock_tool_call = Mock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function = Mock()
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"city": "Tokyo"}'
    mock_choice.message.tool_calls = [mock_tool_call]

    mock_response.choices = [mock_choice]
    mock_response.usage = None

    # Collect chunks
    chunks = []
    async for chunk in handler._handle_non_streaming(mock_response):
        chunks.append(chunk)

    # Verify results
    assert len(chunks) == 1  # only message

    message_chunk = chunks[0]
    assert isinstance(message_chunk, AssistantMessageEvent)
    assert len(message_chunk.message.content) == 1
    assert isinstance(message_chunk.message.content[0], AssistantToolCall)
    assert message_chunk.message.content[0].call_id == "call_123"
    assert message_chunk.message.content[0].name == "get_weather"
    assert message_chunk.message.content[0].arguments == '{"city": "Tokyo"}'


@pytest.mark.asyncio
async def test_completion_handler_no_choices():
    """Test completion handler with no choices."""
    handler = CompletionResponseHandler()

    mock_response = Mock()
    mock_response.choices = []
    mock_response.usage = None

    chunks = []
    async for chunk in handler._handle_non_streaming(mock_response):
        chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_completion_handler_streaming_invalid_response():
    """Test completion handler streaming with invalid response type."""
    handler = CompletionResponseHandler()

    # Mock invalid response (no async iteration support)
    mock_response = Mock()

    with pytest.raises(TypeError, match="Response does not support async iteration"):
        async for _chunk in handler._handle_streaming(mock_response):
            pass


@pytest.mark.asyncio
async def test_responses_handler_non_streaming_with_text():
    """Test responses handler with non-streaming text response."""
    handler = ResponsesAPIHandler()

    # Mock response
    mock_response = Mock()
    mock_response.model = "gpt-4"

    # Mock output item with text content
    mock_output_item = Mock()
    mock_output_item.type = "text"
    mock_content_item = Mock()
    mock_content_item.text = "Hello from responses API!"
    mock_output_item.content = [mock_content_item]

    mock_response.output = [mock_output_item]

    # Mock usage
    mock_usage = Mock()
    mock_usage.input_tokens = 15
    mock_usage.output_tokens = 8
    mock_response.usage = mock_usage

    # Collect chunks
    chunks = []
    async for chunk in handler._handle_non_streaming(mock_response):
        chunks.append(chunk)

    # Verify results
    assert len(chunks) == 2  # message + usage

    # Check message chunk
    message_chunk = chunks[0]
    assert isinstance(message_chunk, AssistantMessageEvent)
    assert len(message_chunk.message.content) == 1
    assert isinstance(message_chunk.message.content[0], AssistantTextContent)
    assert message_chunk.message.content[0].text == "Hello from responses API!"

    # Check usage chunk
    usage_chunk = chunks[1]
    assert isinstance(usage_chunk, UsageEvent)
    assert usage_chunk.usage.input_tokens == 15
    assert usage_chunk.usage.output_tokens == 8


@pytest.mark.asyncio
async def test_responses_handler_non_streaming_with_function_call():
    """Test responses handler with function call."""
    handler = ResponsesAPIHandler()

    # Mock response with function call
    mock_response = Mock()
    mock_response.model = "gpt-4"

    # Mock function call output item
    mock_output_item = Mock()
    mock_output_item.type = "function_call"
    mock_output_item.call_id = "call_456"
    mock_output_item.name = "search_web"
    mock_output_item.arguments = {"query": "Python"}

    mock_response.output = [mock_output_item]
    mock_response.usage = None

    # Collect chunks
    chunks = []
    async for chunk in handler._handle_non_streaming(mock_response):
        chunks.append(chunk)

    # Verify results
    assert len(chunks) == 1  # only message

    message_chunk = chunks[0]
    assert isinstance(message_chunk, AssistantMessageEvent)
    assert len(message_chunk.message.content) == 1
    assert isinstance(message_chunk.message.content[0], AssistantToolCall)
    assert message_chunk.message.content[0].call_id == "call_456"
    assert message_chunk.message.content[0].name == "search_web"
    assert message_chunk.message.content[0].arguments == {"query": "Python"}


@pytest.mark.asyncio
async def test_responses_handler_no_output():
    """Test responses handler with no output."""
    handler = ResponsesAPIHandler()

    mock_response = Mock()
    mock_response.output = []
    mock_response.usage = None

    chunks = []
    async for chunk in handler._handle_non_streaming(mock_response):
        chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_responses_handler_streaming():
    """Test responses handler streaming (delegates to stream handler)."""
    handler = ResponsesAPIHandler()

    # Mock the stream handler to return some chunks
    with patch("lite_agent.response_handlers.responses.openai_response_stream_handler") as mock_stream:
        mock_chunks = [Mock(), Mock()]

        async def async_gen() -> AsyncGenerator[Mock, None]:
            for chunk in mock_chunks:
                yield chunk

        mock_stream.return_value = async_gen()

        class AsyncStream:
            def __aiter__(self) -> AsyncGenerator[None, None]:
                async def gen() -> AsyncGenerator[None, None]:
                    if False:
                        yield None

                return gen()

            async def aclose(self) -> None:
                return None

        response_stream = AsyncStream()
        chunks = []
        async for chunk in handler._handle_streaming(response_stream):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks == mock_chunks


@pytest.mark.asyncio
async def test_completion_handler_streaming():
    """Test completion handler streaming with async iterable response."""
    handler = CompletionResponseHandler()

    class AsyncStream:
        def __aiter__(self) -> AsyncGenerator[None, None]:
            async def gen() -> AsyncGenerator[None, None]:
                if False:
                    yield None

            return gen()

        async def aclose(self) -> None:
            return None

    mock_response = AsyncStream()

    # Mock the stream handler
    with patch("lite_agent.response_handlers.completion.openai_completion_stream_handler") as mock_stream:
        mock_chunks = [Mock(), Mock()]

        async def async_gen() -> AsyncGenerator[Mock, None]:
            for chunk in mock_chunks:
                yield chunk

        mock_stream.return_value = async_gen()

        chunks = []
        async for chunk in handler._handle_streaming(mock_response):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks == mock_chunks
