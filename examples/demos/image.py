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


agent = Agent(
    model="gpt-4.1-nano",
    name="Photo Judge Agent",
    instructions=(
        "You are a helpful photo judge agent. You will receive a photo and you need to judge "
        "whether it is a good photo or not. You should rate the photo on a scale of 1 to 10, "
        "where 1 is the worst and 10 is the best. If you think the photo is not good, you "
        "should provide a reason why it is not good."
    ),
)


async def main():
    runner = Runner(agent)

    resp = runner.run(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What do you think about this photo?",
                    },
                    {
                        "type": "input_image",
                        "detail": "low",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Lavant_St._Peter_und_Paul_Hochaltar_01.jpg/1024px-Lavant_St._Peter_und_Paul_Hochaltar_01.jpg",
                    },
                ],
            },
        ],
        includes=["assistant_message", "usage"],
    )
    async for chunk in resp:
        logger.info(chunk)


if __name__ == "__main__":
    asyncio.run(main())
