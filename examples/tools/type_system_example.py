"""
示例：展示如何使用改进的 Runner 类型系统

这个示例展示了如何使用新的类型定义来获得更好的类型提示，
支持 BaseModel、TypedDict 和普通字典。
"""

from lite_agent.types import (
    AssistantMessageDict,
    NewUserMessage,
    UserMessageDict,
    UserTextContent,
)


def example_usage():
    """展示不同的消息类型用法"""

    # 1. 使用字符串输入 (最简单的用法)
    user_input_str: str = "Hello, how are you?"
    print(f"字符串输入: {user_input_str}")

    # 2. 使用 BaseModel 实例 (Pydantic 模型)
    user_message_model = NewUserMessage(
        content=[UserTextContent(text="This is a message using BaseModel")],
    )
    print(f"BaseModel 输入: {user_message_model}")

    # 3. 使用 TypedDict (获得良好的类型提示)
    user_message_dict: UserMessageDict = {
        "role": "user",
        "content": "This is a message using TypedDict",
    }
    print(f"TypedDict 输入: {user_message_dict}")

    # 4. 使用普通字典 (保持向后兼容)
    user_message_plain_dict = {
        "role": "user",
        "content": "This is a message using plain dict",
    }
    print(f"普通字典输入: {user_message_plain_dict}")

    # 5. 使用消息序列
    conversation_messages: list[UserMessageDict | AssistantMessageDict] = [
        {
            "role": "user",
            "content": "Hello!",
        },
        {
            "role": "assistant",
            "content": "Hi there! How can I help you?",
        },
        {
            "role": "user",
            "content": "Can you tell me a joke?",
        },
    ]
    print(f"对话消息序列: {conversation_messages}")

    # 所有这些类型都被 UserInput 类型联合支持，
    # 提供了良好的类型检查和自动完成

    # 示例调用 (注意：这些在实际使用中需要 await)
    # await runner.run(user_input_str)
    # await runner.run(user_message_model)
    # await runner.run(user_message_dict)
    # await runner.run(user_message_plain_dict)
    # await runner.run(conversation_messages)


def demonstrate_type_safety():
    """演示类型安全性和 IDE 支持"""

    # TypedDict 提供了结构化的类型提示
    user_msg: UserMessageDict = {
        "role": "user",  # IDE 会提示这里只能是 "user"
        "content": "Hello",  # IDE 知道这个字段是必需的
    }

    assistant_msg: AssistantMessageDict = {
        "role": "assistant",  # IDE 会提示这里只能是 "assistant"
        "content": "Hi there!",
    }

    # 错误示例（类型检查器会捕获这些错误）:
    # wrong_msg: UserMessageDict = {
    #     "role": "wrong_role",  # 类型错误！
    #     "content": "Hello"
    # }

    # missing_content: UserMessageDict = {
    #     "role": "user"
    #     # 缺少 "content" 字段 - 类型错误！
    # }

    print("类型安全性演示完成")
    return [user_msg, assistant_msg]


if __name__ == "__main__":
    example_usage()
    demonstrate_type_safety()
    print("类型系统示例运行完成!")
