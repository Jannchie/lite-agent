"""
测试 chat_display 模块的功能
"""

from rich.console import Console

from lite_agent.chat_display import build_chat_summary_table, display_chat_history
from lite_agent.types import AgentAssistantMessage, AgentSystemMessage, AgentUserMessage


def test_render_chat_history_empty():
    """测试空消息列表的渲染。"""
    console = Console(force_terminal=False)
    display_chat_history([], console=console)
    # 如果没有异常，测试通过


def test_render_chat_history_basic_messages():
    """测试基本消息类型的渲染。"""
    messages = [
        AgentSystemMessage(role="system", content="You are a helpful assistant."),
        AgentUserMessage(role="user", content="Hello!"),
        AgentAssistantMessage(role="assistant", content="Hi there!"),
    ]

    console = Console(force_terminal=False)
    display_chat_history(messages, console=console)
    # 如果没有异常，测试通过


def test_render_chat_history_dict_messages():
    """测试字典格式消息的渲染。"""
    messages = [
        {"role": "user", "content": "Hello from dict"},
        {"role": "assistant", "content": "Hi from dict"},
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "test_function",
            "arguments": '{"param": "value"}',
            "content": "",
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Function result",
        },
    ]

    console = Console(force_terminal=False)
    display_chat_history(messages, console=console)
    # 如果没有异常，测试通过


def test_render_chat_history_with_options():
    """测试不同渲染选项。"""
    messages = [
        AgentUserMessage(role="user", content="This is a very long message that will be displayed in full without truncation"),
        AgentAssistantMessage(role="assistant", content="Short response"),
    ]

    console = Console(force_terminal=False)

    # 测试不同选项 - 移除了 max_content_length 参数，因为现在永远显示完整内容
    display_chat_history(
        messages,
        console=console,
        show_timestamps=False,
        show_indices=False,
        chat_width=60,  # 使用 chat_width 参数替代 max_content_length
    )
    # 如果没有异常，测试通过


def test_create_chat_summary_table():
    """测试聊天摘要表格创建。"""
    messages = [
        AgentUserMessage(role="user", content="Hello"),
        AgentAssistantMessage(role="assistant", content="Hi"),
        {"role": "system", "content": "System message"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "test_func",
            "arguments": "{}",
            "content": "",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "result",
        },
    ]

    table = build_chat_summary_table(messages)
    assert table.title == "Chat Summary"
    # 表格应该被成功创建，没有异常


def test_create_chat_summary_table_empty():
    """测试空消息列表的摘要表格。"""
    table = build_chat_summary_table([])
    assert table.title == "Chat Summary"
    # 即使是空列表，也应该能创建表格


if __name__ == "__main__":
    # 运行简单的手动测试
    print("Running manual tests...")

    test_render_chat_history_empty()
    print("✓ Empty messages test passed")

    test_render_chat_history_basic_messages()
    print("✓ Basic messages test passed")

    test_render_chat_history_dict_messages()
    print("✓ Dict messages test passed")

    test_render_chat_history_with_options()
    print("✓ Options test passed")

    test_create_chat_summary_table()
    print("✓ Summary table test passed")

    test_create_chat_summary_table_empty()
    print("✓ Empty summary table test passed")

    print("\nAll tests passed! 🎉")
