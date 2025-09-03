"""Context utilities for injecting history messages into tool calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from lite_agent.types import NewMessage

T = TypeVar("T")


class HistoryContext(BaseModel, Generic[T]):
    """包含历史消息的上下文容器

    这个类会自动被 Runner 用来包装用户的 context，确保工具函数能够访问历史消息。

    Attributes:
        history_messages: 工具调用前的所有历史消息
        data: 用户自定义的上下文数据（可选）

    Examples:
        >>> # 只需要历史消息的工具
        >>> async def count_messages(ctx: Context[HistoryContext[None]]) -> str:
        ...     return f"总共 {len(ctx.value.history_messages)} 条消息"

        >>> # 需要历史消息和用户数据的工具
        >>> class UserData(BaseModel):
        ...     user_id: str
        >>>
        >>> async def analyze_user(ctx: Context[HistoryContext[UserData]]) -> str:
        ...     messages = ctx.value.history_messages
        ...     user_id = ctx.value.data.user_id
        ...     return f"用户 {user_id} 有 {len(messages)} 条消息"
    """

    history_messages: list[NewMessage]
    data: T | None = None
