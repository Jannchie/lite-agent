#!/usr/bin/env python3
"""测试聊天气泡效果的简单脚本"""

from lite_agent.rich_helpers import render_chat_history
from lite_agent.types import (
    AgentUserMessage,
    AgentAssistantMessage,
    AgentSystemMessage,
    AgentFunctionToolCallMessage,
    AgentFunctionCallOutput,
)

# 创建一些测试消息
messages = [
    AgentSystemMessage(role="system", content="你是一个有用的AI助手。"),
    
    AgentUserMessage(role="user", content="你好！请告诉我今天的天气如何？"),
    
    AgentAssistantMessage(role="assistant", content="你好！我需要调用天气API来获取今天的天气信息。"),
    
    AgentFunctionToolCallMessage(
        type="function_call",
        function_call_id="call_123",
        name="get_weather",
        arguments='{"location": "北京", "date": "今天"}',
        content=""
    ),
    
    AgentFunctionCallOutput(
        type="function_call_output",
        call_id="call_123",
        output="今天北京天气晴朗，温度25°C，微风。"
    ),
    
    AgentAssistantMessage(role="assistant", content="根据天气API的数据，今天北京天气晴朗，温度25°C，有微风。是个非常适合外出的好天气！你有什么户外活动计划吗？"),
    
    AgentUserMessage(role="user", content="太好了！我计划去公园散步。顺便问一下，你能推荐一些适合在公园做的活动吗？这是一条比较长的消息，让我们看看完整的内容是如何显示的，不会被截断。"),
    
    AgentAssistantMessage(role="assistant", content="当然可以！以下是一些适合在公园进行的活动：\n\n1. 慢跑或快走 - 很好的有氧运动\n2. 瑜伽或太极 - 在自然环境中放松身心\n3. 野餐 - 和家人朋友一起享受户外时光\n4. 观鸟 - 欣赏和识别不同的鸟类\n5. 摄影 - 捕捉美丽的自然景色\n6. 阅读 - 在树荫下安静地看书\n\n希望这些建议对你有帮助！"),
]

print("🎨 展示聊天气泡效果:")
print("=" * 60)

# 渲染聊天记录，展示气泡效果
render_chat_history(messages, chat_width=60)

print("\n" + "=" * 60)
print("✨ 注意观察:")
print("- 👤 用户消息（蓝色）靠左显示")
print("- 🤖 助手消息（绿色）靠右显示") 
print("- ⚙️ 系统消息（黄色）居中显示")
print("- 🛠️ 函数调用（紫色）居中显示")
print("- 📤 函数输出（青色）居中显示")
print("- 📝 所有消息内容完整显示，不截断")
