"""
简单使用示例：Rich 聊天记录渲染
"""

from lite_agent import display_chat_history, display_chat_summary
from lite_agent.types import AgentAssistantMessage, AgentUserMessage

# 创建一些示例消息
messages = [
    AgentUserMessage(role="user", content="你好!"),
    AgentAssistantMessage(role="assistant", content="你好!我是 AI 助手,很高兴为您服务。"),
    {"role": "user", "content": "今天天气怎么样?"},
    {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city": "北京"}',
        "content": "",
    },
    {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "北京今天晴朗,温度 20°C",
    },
    {"role": "assistant", "content": "北京今天天气很好,晴朗,温度 20°C。"},
]

# 显示聊天摘要
print("📊 聊天摘要:")
display_chat_summary(messages)

print("\n" + "=" * 60)
print("💬 详细聊天记录:")
print("=" * 60)

# 渲染完整聊天记录
display_chat_history(messages)

print("\n🎛️ 其他渲染选项:")
display_chat_history(
    messages,
    show_timestamps=False,
    show_indices=False,
)
