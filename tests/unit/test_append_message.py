"""
单独的 append_message 方法测试文件
专门测试 Runner.append_message 方法的各种用例
"""

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import (
    AgentAssistantMessage,
    AgentSystemMessage,
    AgentUserMessage,
    AssistantTextContent,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    UserTextContent,
)


class DummyAgent(Agent):
    """用于测试的虚拟 Agent"""

    def __init__(self) -> None:
        super().__init__(model="dummy-model", name="Dummy Agent", instructions="This is a dummy agent for testing.")


class TestAppendMessage:
    """Runner.append_message 方法的测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.runner = Runner(agent=DummyAgent())

    def test_append_message_with_user_message_object(self):
        """测试使用 AgentUserMessage 对象添加消息"""
        user_message = AgentUserMessage(role="user", content="Hello, how are you?")

        self.runner.append_message(user_message)

        assert len(self.runner.messages) == 1
        # Now expects NewUserMessage since append_message converts to new format
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].role == "user"
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "Hello, how are you?"

    def test_append_message_with_assistant_message_object(self):
        """测试使用 AgentAssistantMessage 对象添加消息"""
        assistant_message = AgentAssistantMessage(role="assistant", content="I'm doing well, thank you!")

        self.runner.append_message(assistant_message)

        assert len(self.runner.messages) == 1
        # Now expects NewAssistantMessage since append_message converts to new format
        assert isinstance(self.runner.messages[0], NewAssistantMessage)
        assert self.runner.messages[0].role == "assistant"
        assert isinstance(self.runner.messages[0].content[0], AssistantTextContent)
        assert self.runner.messages[0].content[0].text == "I'm doing well, thank you!"

    def test_append_message_with_system_message_object(self):
        """测试使用 AgentSystemMessage 对象添加消息"""
        system_message = AgentSystemMessage(role="system", content="You are a helpful assistant.")

        self.runner.append_message(system_message)

        assert len(self.runner.messages) == 1
        # Now expects NewSystemMessage since append_message converts to new format
        assert isinstance(self.runner.messages[0], NewSystemMessage)
        assert self.runner.messages[0].role == "system"
        assert self.runner.messages[0].content == "You are a helpful assistant."

    def test_append_message_with_user_dict(self):
        """测试dict格式被正确转换为NewUserMessage"""
        user_dict = {"role": "user", "content": "Hello from dict!"}

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].role == "user"
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "Hello from dict!"

    def test_append_message_with_assistant_dict(self):
        """测试dict格式被正确转换为NewAssistantMessage"""
        assistant_dict = {"role": "assistant", "content": "Hello from assistant dict!"}

        self.runner.append_message(assistant_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewAssistantMessage)
        assert self.runner.messages[0].role == "assistant"
        assert isinstance(self.runner.messages[0].content[0], AssistantTextContent)
        assert self.runner.messages[0].content[0].text == "Hello from assistant dict!"

    def test_append_message_with_system_dict(self):
        """测试dict格式被正确转换为NewSystemMessage"""
        system_dict = {"role": "system", "content": "System message from dict"}

        self.runner.append_message(system_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewSystemMessage)
        assert self.runner.messages[0].role == "system"
        assert self.runner.messages[0].content == "System message from dict"

    def test_append_message_with_dict_missing_role(self):
        """测试缺少role字段的dict会抛出ValueError"""
        invalid_dict = {"content": "Missing role field"}

        # Should raise ValueError for missing/invalid role
        with pytest.raises(ValueError, match="Unsupported message role"):
            self.runner.append_message(invalid_dict)

    def test_append_message_multiple_messages(self):
        """测试添加多条消息"""
        from lite_agent.types import NewAssistantMessage, NewSystemMessage, NewUserMessage

        # 添加用户消息 (using legacy format, converted to new)
        user_message = AgentUserMessage(role="user", content="Hello")
        self.runner.append_message(user_message)

        # 添加助手消息 (using new format)
        assistant_message = NewAssistantMessage(content=[AssistantTextContent(text="Hi there!")])
        self.runner.append_message(assistant_message)

        # 添加系统消息 (using legacy format, converted to new)
        system_message = AgentSystemMessage(role="system", content="Be helpful")
        self.runner.append_message(system_message)

        assert len(self.runner.messages) == 3
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].role == "user"
        assert isinstance(self.runner.messages[1], NewAssistantMessage)
        assert self.runner.messages[1].role == "assistant"
        assert isinstance(self.runner.messages[2], NewSystemMessage)
        assert self.runner.messages[2].role == "system"

    def test_append_message_preserves_order(self):
        """测试dict格式消息按顺序正确转换"""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]

        # Add all dict messages and verify they're converted properly
        for msg in messages:
            self.runner.append_message(msg)

        assert len(self.runner.messages) == 3
        assert self.runner.messages[0].content[0].text == "First message"  # type: ignore[union-attr]
        assert self.runner.messages[1].content[0].text == "Second message"  # type: ignore[union-attr]
        assert self.runner.messages[2].content[0].text == "Third message"  # type: ignore[union-attr]

    def test_append_message_with_complex_assistant_dict(self):
        """测试添加包含工具调用的助手消息字典"""

        assistant_dict = {
            "role": "assistant",
            "content": "I'll help you with that.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "New York"}'},
                    "id": "call_123",
                    "index": 0,
                },
            ],
        }

        # Dict should be converted to NewAssistantMessage with tool calls
        self.runner.append_message(assistant_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewAssistantMessage)
        # Check that the message has both text content and tool calls
        assert len(self.runner.messages[0].content) >= 1  # Should have text and/or tool calls

    def test_append_message_empty_content(self):
        """测试空内容的dict消息被正确转换"""
        user_dict = {"role": "user", "content": ""}

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].content[0].text == ""

    def test_append_message_with_extra_fields_in_dict(self):
        """测试字典包含额外字段时的处理"""

        user_dict = {
            "role": "user",
            "content": "Hello",
            "extra_field": "should be ignored",
            "timestamp": "2024-01-01",
        }

        # Dict should be converted, extra fields ignored
        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].content[0].text == "Hello"
        # 额外字段应该被忽略（Pydantic 会过滤未定义的字段）

    def test_runner_messages_initialization(self):
        """测试 Runner 初始化时消息列表为空"""
        assert len(self.runner.messages) == 0
        assert isinstance(self.runner.messages, list)
