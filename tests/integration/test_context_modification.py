import pytest
from funcall import Context
from pydantic import BaseModel

from lite_agent.context import HistoryContext

# Import NewMessage to define it, then fix Pydantic forward reference issue
from lite_agent.types import NewMessage  # noqa: F401

HistoryContext.model_rebuild()


class ModifiableContext(BaseModel):
    counter: int = 0
    message: str = "initial"


async def increment_counter(context: Context[HistoryContext[ModifiableContext]]) -> str:
    """Increment the counter in context by 1."""
    if not context.value.data:
        msg = "Context data must be present."
        raise ValueError(msg)

    context.value.data.counter += 1
    return f"Counter incremented to {context.value.data.counter}"


async def set_message(context: Context[HistoryContext[ModifiableContext]], new_message: str) -> str:
    """Set a new message in the context."""
    if not context.value.data:
        msg = "Context data must be present."
        raise ValueError(msg)

    context.value.data.message = new_message
    return f"Message set to: {new_message}"


async def get_context_values(context: Context[HistoryContext[ModifiableContext]]) -> str:
    """Get current values from context."""
    if not context.value.data:
        msg = "Context data must be present."
        raise ValueError(msg)

    return f"Counter: {context.value.data.counter}, Message: {context.value.data.message}"


@pytest.mark.asyncio
async def test_context_modification_and_persistence():
    """Test that function calls can modify context and changes persist across calls."""

    # Create test context
    test_data = ModifiableContext(counter=0, message="initial")
    history_context = HistoryContext(data=test_data, history_messages=[])
    context = Context(history_context)

    # Test increment counter
    result1 = await increment_counter(context)
    assert "Counter incremented to 1" in result1
    assert test_data.counter == 1
    assert test_data.message == "initial"

    # Test set message
    result2 = await set_message(context, "modified")
    assert "Message set to: modified" in result2
    assert test_data.counter == 1  # unchanged
    assert test_data.message == "modified"

    # Test get values to verify both modifications persist
    result3 = await get_context_values(context)
    assert "Counter: 1, Message: modified" in result3


@pytest.mark.asyncio
async def test_multiple_context_modifications_in_sequence():
    """Test multiple modifications in a single context."""

    test_data = ModifiableContext(counter=0, message="initial")
    history_context = HistoryContext(data=test_data, history_messages=[])
    context = Context(history_context)

    # Multiple increments
    await increment_counter(context)
    await increment_counter(context)
    assert test_data.counter == 2

    # Set message
    await set_message(context, "double_increment")
    assert test_data.message == "double_increment"

    # Verify final state
    result = await get_context_values(context)
    assert "Counter: 2, Message: double_increment" in result


@pytest.mark.asyncio
async def test_context_isolation_between_instances():
    """Test that context changes don't affect other context instances."""

    # Create two separate contexts
    data1 = ModifiableContext(counter=0, message="context1")
    context1 = Context(HistoryContext(data=data1, history_messages=[]))

    data2 = ModifiableContext(counter=10, message="context2")
    context2 = Context(HistoryContext(data=data2, history_messages=[]))

    # Modify context1
    await increment_counter(context1)
    await set_message(context1, "modified1")

    # Modify context2
    await increment_counter(context2)
    await set_message(context2, "modified2")

    # Verify contexts are independent
    assert data1.counter == 1
    assert data1.message == "modified1"

    assert data2.counter == 11  # was 10, incremented by 1
    assert data2.message == "modified2"


@pytest.mark.asyncio
async def test_context_shared_reference():
    """Test that multiple context references to same data share modifications."""

    # Create shared data
    shared_data = ModifiableContext(counter=5, message="shared")

    # Create two contexts pointing to same data
    context1 = Context(HistoryContext(data=shared_data, history_messages=[]))
    context2 = Context(HistoryContext(data=shared_data, history_messages=[]))

    # Modify through context1
    await increment_counter(context1)
    assert shared_data.counter == 6

    # Verify change is visible through context2
    result = await get_context_values(context2)
    assert "Counter: 6" in result

    # Modify through context2
    await set_message(context2, "modified_shared")
    assert shared_data.message == "modified_shared"

    # Verify change is visible through context1
    result = await get_context_values(context1)
    assert "Message: modified_shared" in result
