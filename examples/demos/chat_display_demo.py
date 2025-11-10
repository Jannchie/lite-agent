"""
示例：使用 chat_display 美观显示聊天记录

这个示例展示了如何使用 chat_display 模块中的函数来美观地显示聊天记录。
"""

import asyncio

from lite_agent import display_messages
from lite_agent.agent import Agent
from lite_agent.runner import Runner


def demo_tools():
    """演示工具函数，用于测试函数调用显示。"""

    def get_weather(city: str) -> str:
        """获取指定城市的天气信息。"""
        return f"The weather in {city} is sunny with 25°C"

    def calculate(expression: str) -> str:
        """计算数学表达式。"""
        try:
            result = eval(expression)  # noqa: S307
        except Exception as e:
            return f"Error calculating {expression}: {e}"
        else:
            return f"{expression} = {result}"

    return [get_weather, calculate]


async def create_sample_chat_history() -> Runner:
    """创建一个包含各种消息类型的示例聊天历史。"""
    # 创建 agent
    agent = Agent(
        model="gpt-4o-mini",
        name="DemoAgent",
        instructions="You are a helpful assistant that can provide weather information and perform calculations.",
        tools=demo_tools(),
    )

    # 创建 runner
    runner = Runner(agent=agent)

    # 手动添加一些示例消息来展示不同的消息类型
    runner.append_message({"role": "system", "content": "You are a helpful assistant."})
    runner.append_message({"role": "user", "content": "Hello! Can you help me with some tasks?"})
    runner.append_message({"role": "assistant", "content": "Of course! I'd be happy to help you. What would you like to do?"})
    runner.append_message({"role": "user", "content": "What's the weather like in Tokyo?"})

    # 添加函数调用消息
    runner.append_message(
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city": "Tokyo"}',
            "content": "",
        },
    )

    # 添加函数调用输出
    runner.append_message(
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "The weather in Tokyo is sunny with 25°C",
        },
    )

    runner.append_message({"role": "assistant", "content": "The weather in Tokyo is sunny with a temperature of 25°C. Is there anything else you'd like to know?"})
    runner.append_message({"role": "user", "content": "Can you calculate 25 * 4 + 10?"})

    # 添加另一个函数调用
    runner.append_message(
        {
            "type": "function_call",
            "call_id": "call_456",
            "name": "calculate",
            "arguments": '{"expression": "25 * 4 + 10"}',
            "content": "",
        },
    )

    runner.append_message(
        {
            "type": "function_call_output",
            "call_id": "call_456",
            "output": "25 * 4 + 10 = 110",
        },
    )

    runner.append_message({"role": "assistant", "content": "The calculation 25 * 4 + 10 equals 110."})

    return runner


async def main():
    """主函数：演示 chat_display 的使用。"""
    print("🎨 Chat Display Demo\n")

    # 创建示例聊天历史
    runner = await create_sample_chat_history()

    # print demo
    display_messages(
        runner.messages,
    )


if __name__ == "__main__":
    asyncio.run(main())
