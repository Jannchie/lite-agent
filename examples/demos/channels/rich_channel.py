from rich.console import Console

from lite_agent.types import AgentChunk, ContentDeltaEvent


class RichChannel:
    def __init__(self) -> None:
        self.console = Console()
        self.map = {
            "function_call": self.handle_tool_call,
            "function_call_output": self.handle_tool_call_result,
            "function_call_delta": self.handle_function_call_delta,
            "content_delta": self.handle_content_delta,
            "usage": self.handle_usage,
        }
        self._new_turn = True

    async def handle(self, chunk: AgentChunk):
        handler = self.map.get(chunk.type)
        if handler is None:
            return None
        return await handler(chunk)  # type: ignore

    def new_turn(self):
        print()
        self._new_turn = True

    async def handle_tool_call(self, chunk: AgentChunk):
        print()
        name = getattr(chunk, "name", "<unknown>")
        arguments = getattr(chunk, "arguments", "")
        self.console.print(f"🛠️  [green]{name}[/green]([yellow]{arguments}[/yellow])")

    async def handle_tool_call_result(self, chunk: AgentChunk):
        name = getattr(chunk, "name", "<unknown>")
        content = getattr(chunk, "content", "")
        self.console.print(f"🛠️  [green]{name}[/green] → [yellow]{content}[/yellow]")

    async def handle_function_call_delta(self, chunk: AgentChunk): ...
    async def handle_content_delta(self, chunk: ContentDeltaEvent):
        if self._new_turn:
            self.console.print("🤖 ", end="")
            self._new_turn = False
        print(chunk.delta, end="", flush=True)

    async def handle_usage(self, chunk: AgentChunk):
        if False:
            usage = chunk.usage
            self.console.print(f"In: {usage.prompt_tokens}, Out: {usage.completion_tokens}, Total: {usage.total_tokens}")
