from typing import Literal

from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: str | None = None


class ToolCall(BaseModel):
    type: Literal["function"]
    function: ToolCallFunction
    id: str
    index: int
