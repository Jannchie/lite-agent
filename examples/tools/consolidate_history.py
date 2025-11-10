"""
Test script for the consolidate_history_transfer function.
"""

from lite_agent import consolidate_history_transfer


def test_consolidate_history():
    """Test the consolidate_history_transfer function."""

    # Test with complex conversation history
    messages = [
        {"role": "user", "content": "Hello, I need help with my computer."},
        {"role": "assistant", "content": "I'd be happy to help! What seems to be the problem?"},
        {"role": "user", "content": "My computer is running slowly."},
        {"role": "assistant", "content": "Let me run a diagnostic to check your system."},
        {"type": "function_call", "name": "run_diagnostic", "arguments": '{"system": "windows"}'},
        {"type": "function_call_output", "call_id": "call_456", "output": "High CPU usage detected"},
        {"role": "assistant", "content": "I found the issue - your CPU usage is very high."},
        {"role": "user", "content": "How can I fix this?"},
    ]

    result = consolidate_history_transfer(messages)

    print("Original messages count:", len(messages))
    print("Consolidated messages count:", len(result))
    print("\nConsolidated content:")
    print("=" * 50)
    if result and isinstance(result[0], dict) and "content" in result[0]:
        print(result[0]["content"])
    print("=" * 50)

    # Test with empty messages
    empty_result = consolidate_history_transfer([])
    print(f"\nEmpty messages result: {empty_result}")

    # Test with single message
    single_message = [{"role": "user", "content": "Just a simple test"}]
    single_result = consolidate_history_transfer(single_message)
    print(f"\nSingle message result count: {len(single_result)}")
    if single_result and isinstance(single_result[0], dict) and "content" in single_result[0]:
        print("Single message consolidated content:")
        print(single_result[0]["content"])


if __name__ == "__main__":
    test_consolidate_history()
