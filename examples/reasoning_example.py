import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.runner import Runner

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

    # 1. 在Agent初始化时设置推理参数
    print("1. Agent初始化时设置reasoning_effort:")
    agent_with_reasoning = Agent(
        model="gpt-4o-mini",
        name="推理助手",
        instructions="你是一个深度分析助手，使用仔细的推理来提供全面的分析。",
        reasoning_effort="high",  # 高强度推理
    )
    print(f"   Agent推理努力程度: {agent_with_reasoning.reasoning_effort}")
    print(f"   客户端推理努力程度: {agent_with_reasoning.client.reasoning_effort}")

    # 2. 使用thinking_config进行更精细的控制
    print("\n2. 使用thinking_config进行精细控制:")
    agent_with_thinking = Agent(
        model="claude-3-5-sonnet-20241022",  # Anthropic模型支持thinking
        name="思考助手",
        instructions="你是一个深思熟虑的助手。",
        thinking_config={"type": "enabled", "budget_tokens": 2048},
    )
    print(f"   Agent思考配置: {agent_with_thinking.thinking_config}")
    print(f"   客户端思考配置: {agent_with_thinking.client.thinking_config}")

    # 3. 同时设置reasoning_effort和thinking_config
    print("\n3. 同时设置多种推理参数:")
    agent_full_config = Agent(
        model="o1-mini",  # OpenAI推理模型
        name="全配置推理助手",
        instructions="你是一个高级推理助手。",
        reasoning_effort="medium",
        thinking_config={"type": "enabled", "budget_tokens": 1024},
    )
    print(f"   推理努力程度: {agent_full_config.reasoning_effort}")
    print(f"   思考配置: {agent_full_config.thinking_config}")

    # 4. 演示运行时覆盖推理参数
    print("\n4. 运行时覆盖推理参数:")
    runner = Runner(agent_with_reasoning)
    print("   - Agent默认使用 reasoning_effort='high'")
    print("   - 运行时可通过 agent_kwargs 覆盖:")
    print("     runner.run(query, agent_kwargs={'reasoning_effort': 'minimal'})")

    # 注意：由于没有实际的API密钥，我们不运行真实的API调用
    print("\n✓ 所有推理配置功能已成功设置！")


async def main():
    """主演示函数。"""
    await demo_reasoning_configurations()

    print("\n" + "="*60)
    print("推理配置使用说明:")
    print("="*60)
    print("""
1. reasoning_effort 参数 (OpenAI兼容):
   - "minimal": 最小推理，快速响应
   - "low": 低强度推理
   - "medium": 中等推理（推荐）
   - "high": 高强度推理，更深入分析

2. thinking_config 参数 (Anthropic兼容):
   - {"type": "enabled", "budget_tokens": N}
   - N 可以是 1024, 2048, 4096 等

3. 使用方法:
   a) Agent初始化时设置: Agent(..., reasoning_effort="high")
   b) 运行时覆盖: runner.run(query, agent_kwargs={"reasoning_effort": "low"})

4. 模型兼容性:
   - OpenAI: o1, o3, o4-mini 系列
   - Anthropic: claude-3.5-sonnet 等
   - 其他: 通过LiteLLM自动转换

5. 示例代码:
   ```python
   agent = Agent(
       model="gpt-4o-mini",
       reasoning_effort="medium",
       thinking_config={"type": "enabled", "budget_tokens": 2048}
   )
   ```
    """)


if __name__ == "__main__":
    asyncio.run(main())
