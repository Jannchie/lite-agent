from rich.console import Console

from lite_agent.types import AgentChunk, ContentDeltaChunk


class RichChannel:
    def __init__(self) -> None:
        self.console = Console()
        self.map = {
            "final_message": self.handle_final_message,
            "tool_call": self.handle_tool_call,
            "tool_call_result": self.handle_tool_call_result,
            "tool_call_delta": self.handle_tool_call_delta,
            "content_delta": self.handle_content_delta,
            "usage": self.handle_usage,
        }
        self.new_turn = True

    def handle(self, chunk: AgentChunk):
        handler = self.map[chunk["type"]]
        handler(chunk)

    def handle_final_message(self, _chunk: AgentChunk):
        print()
        self.new_turn = True

    def handle_tool_call(self, chunk: AgentChunk):
        name = chunk.get("name", "<unknown>")
        arguments = chunk.get("arguments", "")
        self.console.print(f"🛠️  [green]{name}[/green]([yellow]{arguments}[/yellow])")

    def handle_tool_call_result(self, chunk: AgentChunk):
        name = chunk.get("name", "<unknown>")
        content = chunk.get("content", "")
        self.console.print(f"🛠️  [green]{name}[/green] → [yellow]{content}[/yellow]")

    def handle_tool_call_delta(self, chunk: AgentChunk): ...
    def handle_content_delta(self, chunk: ContentDeltaChunk):
        if self.new_turn:
            self.console.print("🤖 ", end="")
            self.new_turn = False
        print(chunk["delta"], end="", flush=True)

    def handle_usage(self, chunk: AgentChunk):
        if False:
            usage = chunk["usage"]
            self.console.print(f"In: {usage.prompt_tokens}, Out: {usage.completion_tokens}, Total: {usage.total_tokens}")
