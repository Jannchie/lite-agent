from rich.console import Console

from lite_agent.types import AgentChunk, ContentDeltaChunk


class RichChannel:
    def __init__(self) -> None:
        self.console = Console()
        self.map = {
            "final_message": self.handle_final_message,
            "function_call": self.handle_tool_call,
            "function_call_output": self.handle_tool_call_result,
            "tool_call_delta": self.handle_tool_call_delta,
            "content_delta": self.handle_content_delta,
            "usage": self.handle_usage,
            "require_confirm": self.handle_require_confirm,
        }
        self.new_turn = True

    async def handle(self, chunk: AgentChunk):
        handler = self.map[chunk.type]
        return await handler(chunk)

    async def handle_final_message(self, _chunk: AgentChunk):
        print()
        self.new_turn = True

    async def handle_tool_call(self, chunk: AgentChunk):
        name = getattr(chunk, "name", "<unknown>")
        arguments = getattr(chunk, "arguments", "")
        self.console.print(f"🛠️  [green]{name}[/green]([yellow]{arguments}[/yellow])")

    async def handle_tool_call_result(self, chunk: AgentChunk):
        name = getattr(chunk, "name", "<unknown>")
        content = getattr(chunk, "content", "")
        self.console.print(f"🛠️  [green]{name}[/green] → [yellow]{content}[/yellow]")

    async def handle_tool_call_delta(self, chunk: AgentChunk): ...
    async def handle_content_delta(self, chunk: ContentDeltaChunk):
        if self.new_turn:
            self.console.print("🤖 ", end="")
            self.new_turn = False
        print(chunk.delta, end="", flush=True)

    async def handle_usage(self, chunk: AgentChunk):
        if False:
            usage = chunk.usage
            self.console.print(f"In: {usage.prompt_tokens}, Out: {usage.completion_tokens}, Total: {usage.total_tokens}")

    async def handle_require_confirm(self, chunk: AgentChunk): ...
