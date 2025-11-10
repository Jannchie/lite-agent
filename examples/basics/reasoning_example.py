import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.client import OpenAIClient

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


async def analyze_complex_problem(problem_description: str) -> str:
    """Analyze a complex problem and return insights."""
    return f"Analysis for: {problem_description}\n- Key factors identified\n- Potential solutions outlined\n- Risk assessment completed"


async def demo_reasoning_configurations():
    """演示不同的推理配置方法。"""
    print("=== 推理配置演示 ===\n")

    # 1. 使用 OpenAIClient 设置推理强度（字符串形式）
    print("1. 使用 OpenAIClient 设置推理强度:")
    agent_with_reasoning = Agent(
        model=OpenAIClient(model="gpt-4o-mini", reasoning="high"),
        name="推理助手",
        instructions="你是一个深度分析助手，使用仔细的推理来提供全面的分析。",
    )
    print(f"   客户端推理努力程度: {agent_with_reasoning.client.reasoning_effort}")
    print(f"   客户端思考配置: {agent_with_reasoning.client.thinking_config}")

    # 2. 使用 OpenAIClient 进行更精细的控制（字典形式）
    print("\n2. 使用 OpenAIClient 进行精细控制:")
    agent_with_thinking = Agent(
        model=OpenAIClient(
            model="claude-3-5-sonnet-20241022",  # Anthropic模型支持thinking
            reasoning={"type": "enabled", "budget_tokens": 2048},
        ),
        name="思考助手",
        instructions="你是一个深思熟虑的助手。",
    )
    print(f"   客户端推理努力程度: {agent_with_thinking.client.reasoning_effort}")
    print(f"   客户端思考配置: {agent_with_thinking.client.thinking_config}")

    # 3. 使用 {"effort": "value"} 格式（推荐）
    print('\n3. 使用 {"effort": "value"} 格式:')
    agent_effort_reasoning = Agent(
        model=OpenAIClient(
            model="o1-mini",  # OpenAI推理模型
            reasoning={"effort": "medium"},
        ),
        name="努力推理助手",
        instructions="你是一个高级推理助手。",
    )
    print(f"   客户端推理努力程度: {agent_effort_reasoning.client.reasoning_effort}")
    print(f"   客户端思考配置: {agent_effort_reasoning.client.thinking_config}")

    # 4. 使用布尔值设置推理（会默认使用medium级别）
    print("\n4. 使用布尔值启用推理:")
    agent_bool_reasoning = Agent(
        model=OpenAIClient(model="o1-mini", reasoning=True),  # 布尔值，会使用默认的medium级别
        name="布尔推理助手",
        instructions="你是一个高级推理助手。",
    )
    print(f"   客户端推理努力程度: {agent_bool_reasoning.client.reasoning_effort}")
    print(f"   客户端思考配置: {agent_bool_reasoning.client.thinking_config}")

    # 5. 演示运行时推理参数配置
    print("\n5. 运行时推理参数配置:")
    print("   - 推理配置现在只能在 OpenAIClient 初始化时设置")
    print("   - 如需动态调整，请创建不同的 Agent 实例")

    # 注意：由于没有实际的API密钥，我们不运行真实的API调用
    print("\n✓ 所有推理配置功能已成功设置！")


async def main():
    """主演示函数。"""
    await demo_reasoning_configurations()

    print("\n" + "=" * 60)
    print("推理配置使用说明:")
    print("=" * 60)
    print("""
1. reasoning 参数类型 (在 OpenAIClient 中设置):
   - 字符串: "minimal", "low", "medium", "high" -> reasoning_effort
   - {"effort": "value"}: {"effort": "minimal"} -> reasoning_effort (推荐)
   - 字典: {"type": "enabled", "budget_tokens": N} -> thinking_config
   - 布尔: True -> "medium", False -> 不启用

2. 使用方法:
   ```python
   # 方法1: 字符串形式
   agent = Agent(
       model=OpenAIClient(model="gpt-4o-mini", reasoning="high")
   )

   # 方法2: {"effort": "value"} 形式（推荐）
   agent = Agent(
       model=OpenAIClient(model="gpt-4o-mini", reasoning={"effort": "minimal"})
   )

   # 方法3: 字典形式 (Anthropic模型)
   agent = Agent(
       model=OpenAIClient(
           model="claude-3-5-sonnet-20241022",
           reasoning={"type": "enabled", "budget_tokens": 2048}
       )
   )
   ```

3. 模型兼容性:
   - OpenAI: o1, o3, gpt-4o-mini 等支持 reasoning_effort
   - Anthropic: claude-3.5-sonnet 等支持 thinking
   - 其他: 通过 LiteLLM 自动转换
    """)


if __name__ == "__main__":
    asyncio.run(main())
