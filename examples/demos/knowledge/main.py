import asyncio
import logging

from rich.logging import RichHandler

from lite_agent.agent import Agent
from lite_agent.chat_display import display_messages
from lite_agent.client import OpenAIClient
from lite_agent.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("lite_agent")
logger.setLevel(logging.DEBUG)


def list_knowledges() -> list[str]:
    """
    列出系统中存在的知识，返回的是知识的标题
    """
    return ["苹果的价格的计算方法", "香蕉的价格", "茄子的价格", "芒果的价格", "其他水果蔬菜的价格"]


def read_knowledge(knowledge: str) -> str:
    """
    通过知识的标题获取其内容
    """
    match knowledge:
        case "苹果的价格的计算方法":
            return "苹果的价格是两倍的香蕉的价格减去茄子的价格"
        case "香蕉的价格":
            return "香蕉的价格为3元"
        case "其他水果蔬菜的价格":
            return "茄子的价格为5元"
        case "茄子的价格":
            return "茄子的价格请参阅其他知识库"
        case "芒果的价格":
            return "芒果的价格为7元"
        case _:
            return "不存在的知识"


agent = Agent(
    model=OpenAIClient(model="gpt-5-nano", reasoning={"effort": "minimal"}),
    name="Assistant",
    instructions="你是一个有帮助的助手。",
    tools=[list_knowledges, read_knowledge],
)


async def main():
    runner = Runner(agent, streaming=True)
    await runner.run_until_complete(
        "苹果的价格是多少？仔细检查知识库告诉我答案。",
        includes=["usage", "assistant_message", "function_call", "function_call_output", "timing"],
    )

    display_messages(runner.messages)


if __name__ == "__main__":
    asyncio.run(main())
