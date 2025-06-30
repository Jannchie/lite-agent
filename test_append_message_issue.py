#!/usr/bin/env python3
"""测试 append_message 方法的问题演示"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / ".." / "src"))

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import AgentFunctionCallOutput, AgentFunctionToolCallMessage, AgentUserMessage


def test_append_message_issue():
    """演示 append_message 方法的奇怪问题"""

    print("=" * 60)
    print("测试 append_message 方法问题")
    print("=" * 60)

    # 创建 agents
    parent = Agent(model="gpt-4.1", name="ParentAgent", instructions="Test agent")
    child = Agent(model="gpt-4.1", name="ChildAgent", instructions="Test child agent")
    parent.add_handoff(child)

    runner = Runner(parent)

    print(f"初始状态: {len(runner.messages)} 条消息")

    # 1. 测试 dict 消息 - 这个应该工作
    print("\n--- 测试 1: Dict 消息 ---")
    dict_msg = {"role": "user", "content": "Hello"}
    print(f"要添加的消息: {dict_msg}")
    print(f"Messages before: {len(runner.messages)}")
    runner.append_message(dict_msg)
    print(f"Messages after: {len(runner.messages)}")  # 应该是 1

    # 2. 测试 AgentFunctionToolCallMessage - 这里有问题
    print("\n--- 测试 2: AgentFunctionToolCallMessage ---")
    transfer_message = AgentFunctionToolCallMessage(
        type="function_call",
        function_call_id="call_1",
        name="transfer_to_agent",
        arguments='{"name": "ChildAgent"}',
        content="",
    )

    print(f"要添加的消息类型: {type(transfer_message)}")
    print(f"消息内容: {transfer_message}")

    # 检查 isinstance
    from lite_agent.types import AgentAssistantMessage, AgentSystemMessage
    isinstance_check = isinstance(transfer_message, (AgentUserMessage, AgentAssistantMessage, AgentSystemMessage,
                                                   AgentFunctionToolCallMessage, AgentFunctionCallOutput))
    print(f"isinstance 检查结果: {isinstance_check}")

    print(f"Messages before: {len(runner.messages)}")  # 应该是 1
    runner.append_message(transfer_message)  # isinstance 检查通过
    print(f"Messages after: {len(runner.messages)}")   # 奇怪：仍然是 1！

    # 3. 测试手动添加同样的消息
    print("\n--- 测试 3: 手动添加 ---")
    print(f"Messages before manual append: {len(runner.messages)}")
    runner.messages.append(transfer_message)
    print(f"Messages after manual append: {len(runner.messages)}")  # 这个应该是 2

    # 移除手动添加的，重新测试
    runner.messages.pop()
    print(f"Messages after pop: {len(runner.messages)}")

    # 4. 测试 AgentFunctionCallOutput
    print("\n--- 测试 4: AgentFunctionCallOutput ---")
    output_message = AgentFunctionCallOutput(
        type="function_call_output",
        call_id="call_1",
        output="Transferring to agent: ChildAgent",
    )

    print(f"要添加的消息类型: {type(output_message)}")
    print(f"Messages before: {len(runner.messages)}")
    runner.append_message(output_message)
    print(f"Messages after: {len(runner.messages)}")

    # 5. 检查最终的消息内容
    print("\n--- 最终消息列表 ---")
    for i, msg in enumerate(runner.messages):
        print(f"消息 {i}: {type(msg)} - {msg}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

def test_isinstance_details():
    """详细测试 isinstance 检查"""
    print("\n" + "=" * 60)
    print("详细测试 isinstance 检查")
    print("=" * 60)

    transfer_message = AgentFunctionToolCallMessage(
        type="function_call",
        function_call_id="call_1",
        name="transfer_to_agent",
        arguments='{"name": "ChildAgent"}',
        content="",
    )

    from lite_agent.types import AgentAssistantMessage, AgentSystemMessage, RunnerMessage

    print(f"消息类型: {type(transfer_message)}")
    print(f"消息的 MRO: {type(transfer_message).__mro__}")

    print("\n单独的 isinstance 检查:")
    print(f"  isinstance(msg, AgentUserMessage): {isinstance(transfer_message, AgentUserMessage)}")
    print(f"  isinstance(msg, AgentAssistantMessage): {isinstance(transfer_message, AgentAssistantMessage)}")
    print(f"  isinstance(msg, AgentSystemMessage): {isinstance(transfer_message, AgentSystemMessage)}")
    print(f"  isinstance(msg, AgentFunctionToolCallMessage): {isinstance(transfer_message, AgentFunctionToolCallMessage)}")
    print(f"  isinstance(msg, AgentFunctionCallOutput): {isinstance(transfer_message, AgentFunctionCallOutput)}")

    print("\n元组 isinstance 检查:")
    tuple_check = isinstance(transfer_message, (AgentUserMessage, AgentAssistantMessage, AgentSystemMessage,
                                              AgentFunctionToolCallMessage, AgentFunctionCallOutput))
    print(f"  元组检查结果: {tuple_check}")

    print(f"\nRunnerMessage 类型: {RunnerMessage}")

if __name__ == "__main__":
    test_isinstance_details()
    test_append_message_issue()
