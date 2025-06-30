#!/usr/bin/env python3
"""
测试新的消息对齐方式：
- 用户消息：靠右（蓝色）
- 助手消息：靠左（绿色）
- 函数调用：靠左（紫色）
- 函数输出：靠左（青色）
- 系统消息：居中（黄色）
"""

from lite_agent.types import (
    AgentAssistantMessage,
    AgentFunctionCallOutput,
    AgentFunctionToolCallMessage,
    AgentSystemMessage,
    AgentUserMessage,
)
from lite_agent.rich_helpers import render_chat_history

def test_new_alignment():
    """测试新的消息对齐方式"""
    messages = [
        AgentSystemMessage(role="system", content="欢迎使用 lite-agent！这是系统消息，应该居中显示。"),
        
        AgentUserMessage(role="user", content="你好！我想了解一下天气情况。这是用户消息，应该靠右显示。"),
        
        AgentAssistantMessage(role="assistant", content="你好！我很乐意帮助你查询天气信息。让我为你调用天气查询功能。这是助手消息，应该靠左显示。"),
        
        AgentFunctionToolCallMessage(
            type="function_call",
            function_call_id="call_123",
            name="get_weather",
            arguments='{"city": "北京", "unit": "celsius"}',
            content=""
        ),
        
        AgentFunctionCallOutput(
            type="function_call_output",
            call_id="call_123",
            output="北京今天天气晴朗，温度25°C，湿度60%，风力3级。"
        ),
        
        AgentAssistantMessage(role="assistant", content="根据查询结果，北京今天天气很不错！温度适宜，是个出门的好日子。还有什么其他问题需要帮助的吗？"),
        
        AgentUserMessage(role="user", content="谢谢！还想问一下明天的天气预报。"),
        
        AgentFunctionToolCallMessage(
            type="function_call",
            function_call_id="call_124",
            name="get_weather_forecast",
            arguments='{"city": "北京", "days": 1, "detailed": true}',
            content=""
        ),
        
        AgentFunctionCallOutput(
            type="function_call_output",
            call_id="call_124",
            output="北京明天多云转阴，温度22-28°C，有小雨概率30%，建议带伞。"
        ),
        
        AgentAssistantMessage(role="assistant", content="明天天气可能会有变化，建议你带把伞以防万一。如果还有其他问题，随时可以问我！"),
    ]
    
    print("=" * 80)
    print("新的消息对齐方式测试")
    print("用户消息(蓝色)→靠右 | 助手消息(绿色)→靠左 | 函数调用/输出→靠左 | 系统消息→居中")
    print("=" * 80)
    
    render_chat_history(
        messages,
        chat_width=60,
        show_timestamps=False,
        show_indices=True
    )

if __name__ == "__main__":
    test_new_alignment()
