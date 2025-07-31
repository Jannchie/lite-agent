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
        """测试使用字典格式添加用户消息"""
        user_dict = {"role": "user", "content": "Hello from dict!"}

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].role == "user"
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "Hello from dict!"

    def test_append_message_with_assistant_dict(self):
        """测试使用字典格式添加助手消息"""
        from lite_agent.types import NewAssistantMessage

        assistant_dict = {"role": "assistant", "content": "Hello from assistant dict!"}

        self.runner.append_message(assistant_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewAssistantMessage)
        assert self.runner.messages[0].role == "assistant"
        assert isinstance(self.runner.messages[0].content[0], AssistantTextContent)
        assert self.runner.messages[0].content[0].text == "Hello from assistant dict!"

    def test_append_message_with_system_dict(self):
        """测试使用字典格式添加系统消息"""
        from lite_agent.types import NewSystemMessage

        system_dict = {"role": "system", "content": "System message from dict"}

        self.runner.append_message(system_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewSystemMessage)
        assert self.runner.messages[0].role == "system"
        assert self.runner.messages[0].content == "System message from dict"

    def test_append_message_with_dict_missing_role(self):
        """测试字典格式缺少 role 字段时抛出异常"""
        invalid_dict = {"content": "Missing role field"}

        with pytest.raises(ValueError, match="Message must have a 'role' or 'type' field."):
            self.runner.append_message(invalid_dict)

    def test_append_message_multiple_messages(self):
        """测试添加多条消息"""
        from lite_agent.types import NewAssistantMessage, NewSystemMessage, NewUserMessage

        # 添加用户消息
        user_message = AgentUserMessage(role="user", content="Hello")
        self.runner.append_message(user_message)

        # 添加助手消息
        assistant_dict = {"role": "assistant", "content": "Hi there!"}
        self.runner.append_message(assistant_dict)

        # 添加系统消息
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
        """测试消息添加顺序保持正确"""
        from lite_agent.types import NewAssistantMessage, NewUserMessage

        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]

        for msg in messages:
            self.runner.append_message(msg)

        assert len(self.runner.messages) == 3
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "First message"
        assert isinstance(self.runner.messages[1], NewAssistantMessage)
        assert isinstance(self.runner.messages[1].content[0], AssistantTextContent)
        assert self.runner.messages[1].content[0].text == "Second message"
        assert isinstance(self.runner.messages[2], NewUserMessage)
        assert isinstance(self.runner.messages[2].content[0], UserTextContent)
        assert self.runner.messages[2].content[0].text == "Third message"

    def test_append_message_with_complex_assistant_dict(self):
        """测试添加包含工具调用的助手消息字典"""
        from lite_agent.types import NewAssistantMessage

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

        self.runner.append_message(assistant_dict)

        # 在新格式中，应该产生 1 个消息，包含文本和工具调用
        assert len(self.runner.messages) == 1

        # 消息应该是包含文本和工具调用的 NewAssistantMessage
        assert isinstance(self.runner.messages[0], NewAssistantMessage)
        assert self.runner.messages[0].role == "assistant"

        # 检查内容项
        content = self.runner.messages[0].content
        assert len(content) == 2  # 文本 + 工具调用

        # 第一个内容应该是文本
        assert content[0].type == "text"
        assert isinstance(content[0], AssistantTextContent)
        assert content[0].text == "I'll help you with that."

        # 第二个内容应该是工具调用
        assert content[1].type == "tool_call"
        assert content[1].call_id == "call_123"
        assert content[1].name == "get_weather"
        assert content[1].arguments == {"city": "New York"}

    def test_append_message_empty_content(self):
        """测试添加空内容消息"""
        from lite_agent.types import NewUserMessage

        user_dict = {"role": "user", "content": ""}

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].role == "user"
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == ""

    def test_append_message_with_extra_fields_in_dict(self):
        """测试字典包含额外字段时的处理"""
        from lite_agent.types import NewUserMessage

        user_dict = {
            "role": "user",
            "content": "Hello",
            "extra_field": "should be ignored",
            "timestamp": "2024-01-01",
        }

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)
        assert self.runner.messages[0].role == "user"
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "Hello"
        # 额外字段应该被忽略（Pydantic 会过滤未定义的字段）

    def test_runner_messages_initialization(self):
        """测试 Runner 初始化时消息列表为空"""
        assert len(self.runner.messages) == 0
        assert isinstance(self.runner.messages, list)
