"""
单独的 append_message 方法测试文件
专门测试 Runner.append_message 方法的各种用例
"""

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import AgentAssistantMessage, AgentSystemMessage, AgentToolCallMessage, AgentUserMessage


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
        assert self.runner.messages[0] == user_message
        assert isinstance(self.runner.messages[0], AgentUserMessage)
        assert self.runner.messages[0].role == "user"
        assert self.runner.messages[0].content == "Hello, how are you?"

    def test_append_message_with_assistant_message_object(self):
        """测试使用 AgentAssistantMessage 对象添加消息"""
        assistant_message = AgentAssistantMessage(role="assistant", content="I'm doing well, thank you!")

        self.runner.append_message(assistant_message)

        assert len(self.runner.messages) == 1
        assert self.runner.messages[0] == assistant_message
        assert isinstance(self.runner.messages[0], AgentAssistantMessage)
        assert self.runner.messages[0].role == "assistant"
        assert self.runner.messages[0].content == "I'm doing well, thank you!"

    def test_append_message_with_system_message_object(self):
        """测试使用 AgentSystemMessage 对象添加消息"""
        system_message = AgentSystemMessage(role="system", content="You are a helpful assistant.")

        self.runner.append_message(system_message)

        assert len(self.runner.messages) == 1
        assert self.runner.messages[0] == system_message
        assert isinstance(self.runner.messages[0], AgentSystemMessage)
        assert self.runner.messages[0].role == "system"
        assert self.runner.messages[0].content == "You are a helpful assistant."

    def test_append_message_with_tool_call_message_object(self):
        """测试使用 AgentToolCallMessage 对象添加消息"""
        tool_message = AgentToolCallMessage(role="tool", tool_call_id="call_123", content="Tool result")

        self.runner.append_message(tool_message)

        assert len(self.runner.messages) == 1
        assert self.runner.messages[0] == tool_message
        assert isinstance(self.runner.messages[0], AgentToolCallMessage)
        assert self.runner.messages[0].role == "tool"
        assert self.runner.messages[0].tool_call_id == "call_123"
        assert self.runner.messages[0].content == "Tool result"

    def test_append_message_with_user_dict(self):
        """测试使用字典格式添加用户消息"""
        user_dict = {"role": "user", "content": "Hello from dict!"}

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], AgentUserMessage)
        assert self.runner.messages[0].role == "user"
        assert self.runner.messages[0].content == "Hello from dict!"

    def test_append_message_with_assistant_dict(self):
        """测试使用字典格式添加助手消息"""
        assistant_dict = {"role": "assistant", "content": "Hello from assistant dict!"}

        self.runner.append_message(assistant_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], AgentAssistantMessage)
        assert self.runner.messages[0].role == "assistant"
        assert self.runner.messages[0].content == "Hello from assistant dict!"

    def test_append_message_with_system_dict(self):
        """测试使用字典格式添加系统消息"""
        system_dict = {"role": "system", "content": "System message from dict"}

        self.runner.append_message(system_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], AgentSystemMessage)
        assert self.runner.messages[0].role == "system"
        assert self.runner.messages[0].content == "System message from dict"

    def test_append_message_with_tool_dict(self):
        """测试使用字典格式添加工具消息"""
        tool_dict = {"role": "tool", "tool_call_id": "call_456", "content": "Tool result from dict"}

        self.runner.append_message(tool_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], AgentToolCallMessage)
        assert self.runner.messages[0].role == "tool"
        assert self.runner.messages[0].tool_call_id == "call_456"
        assert self.runner.messages[0].content == "Tool result from dict"

    def test_append_message_with_dict_missing_role(self):
        """测试字典格式缺少 role 字段时抛出异常"""
        invalid_dict = {"content": "Missing role field"}

        with pytest.raises(ValueError, match="Message must have a 'role' or 'type' field."):
            self.runner.append_message(invalid_dict)

    def test_append_message_multiple_messages(self):
        """测试添加多条消息"""
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
        assert isinstance(self.runner.messages[0], AgentUserMessage)
        assert self.runner.messages[0].role == "user"
        assert isinstance(self.runner.messages[1], AgentAssistantMessage)
        assert self.runner.messages[1].role == "assistant"
        assert isinstance(self.runner.messages[2], AgentSystemMessage)
        assert self.runner.messages[2].role == "system"

    def test_append_message_preserves_order(self):
        """测试消息添加顺序保持正确"""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]

        for msg in messages:
            self.runner.append_message(msg)

        assert len(self.runner.messages) == 3
        assert isinstance(self.runner.messages[0], AgentUserMessage)
        assert self.runner.messages[0].content == "First message"
        assert isinstance(self.runner.messages[1], AgentAssistantMessage)
        assert self.runner.messages[1].content == "Second message"
        assert isinstance(self.runner.messages[2], AgentUserMessage)
        assert self.runner.messages[2].content == "Third message"

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

        self.runner.append_message(assistant_dict)

        # 在新格式中，应该产生 2 个消息：assistant message + function call message
        assert len(self.runner.messages) == 2

        # 第一个消息应该是不含 tool_calls 的 assistant message
        assert isinstance(self.runner.messages[0], AgentAssistantMessage)
        assert self.runner.messages[0].role == "assistant"
        assert self.runner.messages[0].content == "I'll help you with that."

        # 第二个消息应该是 function call message
        from lite_agent.types import AgentFunctionToolCallMessage
        assert isinstance(self.runner.messages[1], AgentFunctionToolCallMessage)
        assert self.runner.messages[1].type == "function_call"
        assert self.runner.messages[1].function_call_id == "call_123"
        assert self.runner.messages[1].name == "get_weather"
        assert self.runner.messages[1].arguments == '{"city": "New York"}'

    def test_append_message_empty_content(self):
        """测试添加空内容消息"""
        user_dict = {"role": "user", "content": ""}

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], AgentUserMessage)
        assert self.runner.messages[0].role == "user"
        assert self.runner.messages[0].content == ""

    def test_append_message_with_extra_fields_in_dict(self):
        """测试字典包含额外字段时的处理"""
        user_dict = {
            "role": "user",
            "content": "Hello",
            "extra_field": "should be ignored",
            "timestamp": "2024-01-01",
        }

        self.runner.append_message(user_dict)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], AgentUserMessage)
        assert self.runner.messages[0].role == "user"
        assert self.runner.messages[0].content == "Hello"
        # 额外字段应该被忽略（Pydantic 会过滤未定义的字段）

    def test_runner_messages_initialization(self):
        """测试 Runner 初始化时消息列表为空"""
        assert len(self.runner.messages) == 0
        assert isinstance(self.runner.messages, list)
