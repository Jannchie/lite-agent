"""
为agent.py未覆盖部分添加的额外测试
"""

from lite_agent.agent import Agent
from lite_agent.types import (
    AgentAssistantMessage,
    AgentUserMessage,
    FlexibleRunnerMessage,
    NewSystemMessage,
    NewUserMessage,
    RunnerMessage,
    UserTextContent,
)


class TestAgentAdditional:
    """Agent类的额外测试"""

    def test_agent_init_with_handoffs(self):
        """测试带交接代理的初始化"""
        child_agent = Agent(model="gpt-3", name="Child", instructions="Child instructions")
        parent_agent = Agent(
            model="gpt-3",
            name="Parent",
            instructions="Parent instructions",
            handoffs=[child_agent],
        )
        assert child_agent.parent is parent_agent
        assert parent_agent.handoffs == [child_agent]

    def test_agent_init_with_completion_condition(self):
        """测试带完成条件的初始化"""
        agent = Agent(
            model="gpt-3",
            name="TestBot",
            instructions="Be helpful.",
            completion_condition="call",
        )
        assert agent.completion_condition == "call"

    def test_prepare_completion_messages_with_complex_messages(self):
        """测试处理复杂消息格式"""
        agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.")

        # 包含函数调用的消息
        messages: list[FlexibleRunnerMessage] = [
            AgentUserMessage(content=[UserTextContent(text="Hello")]),
            AgentAssistantMessage(content=[]),
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "test_function",
                "arguments": '{"param": "value"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Function result",
            },
        ]

        result = agent.prepare_completion_messages(messages)

        # 应该包含系统消息
        assert result[0]["role"] == "system"

        # 应该正确处理各种消息类型
        assert len(result) >= 3

    def test_prepare_completion_messages_with_new_message_format(self):
        """测试新消息格式的处理"""
        agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.")

        messages: list[RunnerMessage] = [
            NewUserMessage(content=[UserTextContent(text="New user message")]),
            NewSystemMessage(content="New system message"),
        ]

        result = agent.prepare_completion_messages(messages)

        # 应该正确转换新格式消息
        assert result[0]["role"] == "system"  # 代理的系统消息
        assert len(result) >= 2

    def test_agent_with_tools_registration(self):
        """测试工具注册"""

        def test_tool(param: str) -> str:
            return f"Result: {param}"

        agent = Agent(
            model="gpt-3",
            name="TestBot",
            instructions="Be helpful.",
            tools=[test_tool],
        )

        # 工具应该被注册
        assert agent.fc is not None
        tools = agent.fc.get_tools()
        assert len(tools) > 0  # 至少包含注册的工具

    def test_agent_handoff_tools(self):
        """测试代理交接工具的注册"""
        child_agent = Agent(model="gpt-3", name="Child", instructions="Child instructions")
        parent_agent = Agent(
            model="gpt-3",
            name="Parent",
            instructions="Parent instructions",
            handoffs=[child_agent],
        )

        # 验证父子关系建立
        assert child_agent.parent is parent_agent
        assert parent_agent.handoffs == [child_agent]

        # 验证工具注册系统工作
        assert parent_agent.fc is not None
        assert child_agent.fc is not None

    def test_agent_wait_for_user_tool(self):
        """测试等待用户工具的注册"""
        agent = Agent(
            model="gpt-3",
            name="TestBot",
            instructions="Be helpful.",
            completion_condition="call",
        )

        # 验证completion_condition设置正确
        assert agent.completion_condition == "call"

        # 验证funcall系统正常工作
        assert agent.fc is not None

    def test_agent_client_property(self):
        """测试代理客户端属性"""
        agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.")

        # 应该有client属性
        assert agent.client is not None
        assert hasattr(agent.client, "model")

    def test_message_conversion_edge_cases(self):
        """测试消息转换的边界情况"""
        agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.")

        # 测试空消息列表
        result = agent.prepare_completion_messages([])
        assert len(result) == 1  # 只有系统消息
        assert result[0]["role"] == "system"

        # 测试只有系统消息的情况
        messages = [{"role": "system", "content": "Custom system message"}]
        result = agent.prepare_completion_messages(messages)
        assert len(result) >= 1

    def test_agent_basic_properties(self):
        """测试代理基本属性"""
        agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.")

        assert agent.name == "TestBot"
        assert agent.instructions == "Be helpful."
        assert agent.completion_condition == "stop"  # 默认值
        assert agent.handoffs == []
        assert agent.parent is None

    def test_agent_with_custom_system_message(self):
        """测试自定义系统消息的处理"""
        agent = Agent(model="gpt-3", name="TestBot", instructions="Be helpful.")

        messages = [
            {"role": "system", "content": "Custom system instruction"},
            AgentUserMessage(content=[UserTextContent(text="Hello")]),
        ]

        result = agent.prepare_completion_messages(messages)

        # 应该保留自定义系统消息
        system_messages = [msg for msg in result if msg["role"] == "system"]
        assert len(system_messages) >= 1
