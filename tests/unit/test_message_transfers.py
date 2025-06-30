"""Test the predefined message transfer functions."""

from lite_agent.message_transfers import consolidate_history_transfer


def test_consolidate_history_transfer_basic():
    """Test basic functionality of consolidate_history_transfer."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["role"] == "user"  # type: ignore
    assert "以下是目前发生的所有交互:" in result[0]["content"]  # type: ignore
    assert "<conversation_history>" in result[0]["content"]  # type: ignore
    assert "<message role='user'>Hello</message>" in result[0]["content"]  # type: ignore
    assert "<message role='assistant'>Hi there!</message>" in result[0]["content"]  # type: ignore
    assert "接下来该做什么?" in result[0]["content"]  # type: ignore


def test_consolidate_history_transfer_with_function_calls():
    """Test consolidate_history_transfer with function calls."""
    messages = [
        {"role": "user", "content": "Check the weather"},
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "Tokyo"}'},
        {"type": "function_call_output", "call_id": "call_123", "output": "Sunny, 22°C"},
        {"role": "assistant", "content": "The weather in Tokyo is sunny and 22°C."},
    ]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    content = result[0]["content"]  # type: ignore
    assert "<function_call name='get_weather' arguments='{\"city\": \"Tokyo\"}' />" in content
    assert "<function_result call_id='call_123'>Sunny, 22°C</function_result>" in content


def test_consolidate_history_transfer_empty():
    """Test consolidate_history_transfer with empty messages."""
    result = consolidate_history_transfer([])
    assert result == []


def test_consolidate_history_transfer_single_message():
    """Test consolidate_history_transfer with a single message."""
    messages = [{"role": "user", "content": "Test message"}]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    assert "<message role='user'>Test message</message>" in result[0]["content"]  # type: ignore


def test_consolidate_history_transfer_mixed_types():
    """Test consolidate_history_transfer with mixed message types."""
    from lite_agent.types import AgentUserMessage

    messages = [
        AgentUserMessage(role="user", content="Pydantic message"),
        {"role": "assistant", "content": "Dict message"},
        {"type": "function_call", "name": "test_func", "arguments": "{}"},
    ]

    result = consolidate_history_transfer(messages)

    assert len(result) == 1
    content = result[0]["content"]  # type: ignore
    assert "<message role='user'>Pydantic message</message>" in content
    assert "<message role='assistant'>Dict message</message>" in content
    assert "<function_call name='test_func' arguments='{}' />" in content
