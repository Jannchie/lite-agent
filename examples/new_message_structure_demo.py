"""
演示新的消息结构
这个示例展示了如何使用新的结构化消息类型以及与旧格式的转换
"""

from datetime import datetime, timezone

from lite_agent.types import (
    AgentAssistantMessage,
    AgentUserMessage,
    AssistantMessageMeta,
    AssistantTextContent,
    AssistantToolCall,
    AssistantToolCallResult,
    BasicMessageMeta,
    MessageUsage,
    NewAssistantMessage,
    NewSystemMessage,
    NewUserMessage,
    UserImageContent,
    UserTextContent,
    assistant_message_to_llm_dict,
)


def demonstrate_new_structure():
    """演示新的消息结构如何表示对话"""
    print("=== 新的消息结构演示 ===\n")

    # 创建用户消息（支持多种内容类型）
    user_message = NewUserMessage(
        content=[
            UserTextContent(text="纽约的天气怎么样？顺便，看看这张图片："),
            UserImageContent(image_url="https://example.com/nyc.jpg"),
        ],
    )

    # 创建系统消息
    system_message = NewSystemMessage(
        content="你是一个有用的天气助手。",
    )

    # 创建助手消息（包含文本、工具调用和结果）
    usage = MessageUsage(input_tokens=50, output_tokens=30, total_tokens=80)
    assistant_meta = AssistantMessageMeta(
        model="gpt-4",
        usage=usage,
        total_time_ms=1500,
        latency_ms=200,
    )

    assistant_message = NewAssistantMessage(
        content=[
            AssistantTextContent(text="我来帮你查询纽约的天气。"),
            AssistantToolCall(
                call_id="call_123",
                name="get_weather",
                arguments={"location": "New York", "units": "celsius"},
            ),
            AssistantToolCallResult(
                call_id="call_123",
                output="纽约当前气温25°C，晴朗。",
                execution_time_ms=150,
            ),
            AssistantTextContent(text="根据查询结果，纽约现在的天气是25°C，天气晴朗！"),
        ],
        meta=assistant_meta,
    )

    # 展示消息结构
    messages = [system_message, user_message, assistant_message]

    for i, message in enumerate(messages, 1):
        print(f"消息 {i}: {message.__class__.__name__}")
        print(f"  角色: {message.role}")

        if hasattr(message, "content") and isinstance(message.content, list):
            print(f"  内容项数量: {len(message.content)}")
            for j, content_item in enumerate(message.content, 1):
                print(f"    项 {j}: {content_item.type}")
                if content_item.type == "text":
                    print(f"      文本: {content_item.text[:50]}...")
                elif content_item.type == "image":
                    print(f"      图像URL: {content_item.image_url}")
                elif content_item.type == "tool_call":
                    print(f"      工具: {content_item.name}")
                    print(f"      参数: {content_item.arguments}")
                elif content_item.type == "tool_call_result":
                    print(f"      结果: {content_item.output}")
        else:
            print(f"  内容: {message.content}")

        if hasattr(message.meta, "usage") and message.meta.usage:
            print(f"  Token使用: 输入={message.meta.usage.input_tokens}, 输出={message.meta.usage.output_tokens}")
            print(f"  模型: {message.meta.model}")
            print(f"  总时间: {message.meta.total_time_ms}ms")

        print(f"  发送时间: {message.meta.sent_at}")
        print()


def demonstrate_conversion():
    """演示新旧格式之间的转换"""
    print("=== 格式转换演示 ===\n")

    now = datetime.now(timezone.utc)

    # 创建旧格式的消息序列
    legacy_messages = [
        AgentUserMessage(
            content="查询天气",
            meta=BasicMessageMeta(sent_at=now),
        ),
        AgentAssistantMessage(
            content="我来为你查询天气。",
            meta=AssistantMessageMeta(
                sent_at=now,
                input_tokens=10,
                output_tokens=8,
                latency_ms=100,
            ),
        ),
        # Assistant message with tool call and result in new format
        NewAssistantMessage(
            content=[
                AssistantToolCall(
                    call_id="call_456",
                    name="get_weather",
                    arguments='{"location": "北京"}',
                ),
                AssistantToolCallResult(
                    call_id="call_456",
                    output="北京今天气温18°C，多云。",
                    execution_time_ms=200,
                ),
            ],
            meta=AssistantMessageMeta(sent_at=now),
        ),
    ]

    print("旧格式消息数量:", len(legacy_messages))
    for msg in legacy_messages:
        print(f"  - {msg.__class__.__name__}")

    # 注意：格式转换功能当前不可用
    print("\n注意：convert_legacy_to_new 和 convert_new_to_legacy 函数尚未实现")


def demonstrate_llm_dict_conversion():
    """演示转换为LLM API格式"""
    print("=== LLM API格式转换演示 ===\n")

    # 创建复杂的助手消息
    assistant_message = NewAssistantMessage(
        content=[
            AssistantTextContent(text="我需要调用工具来帮助你。"),
            AssistantToolCall(
                call_id="call_789",
                name="calculate",
                arguments={"expression": "2 + 2"},
            ),
        ],
    )

    # 转换为LLM API格式
    llm_dict = assistant_message_to_llm_dict(assistant_message)

    print("转换为LLM API格式:")
    print(f"  角色: {llm_dict['role']}")
    print(f"  内容: {llm_dict['content']}")
    if "tool_calls" in llm_dict:
        print(f"  工具调用数量: {len(llm_dict['tool_calls'])}")
        for tool_call in llm_dict["tool_calls"]:
            print(f"    ID: {tool_call['id']}")
            print(f"    函数: {tool_call['function']['name']}")
            print(f"    参数: {tool_call['function']['arguments']}")


if __name__ == "__main__":
    demonstrate_new_structure()
    demonstrate_conversion()
    demonstrate_llm_dict_conversion()
