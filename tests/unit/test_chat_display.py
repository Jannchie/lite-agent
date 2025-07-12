"""
测试 chat_display 模块的功能
"""

from lite_agent.chat_display import build_chat_summary_table
from lite_agent.types import AgentAssistantMessage, AgentUserMessage


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
