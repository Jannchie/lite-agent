import abc
import os
from collections.abc import AsyncGenerator
from typing import Any, Literal

import litellm
from litellm.types.llms.openai import ResponsesAPIStreamingResponse
from openai.types.chat import ChatCompletionToolParam
from openai.types.responses import FunctionToolParam


class BaseLLMClient(abc.ABC):
    """Base class for LLM clients."""

    def __init__(self, *, model: str, api_key: str | None = None, api_base: str | None = None, api_version: str | None = None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

    @abc.abstractmethod
    async def completion(self, messages: list[Any], tools: list[ChatCompletionToolParam] | None = None, tool_choice: str = "auto") -> Any:  # noqa: ANN401
        """Perform a completion request to the LLM."""

    @abc.abstractmethod
    async def responses(
        self,
        messages: list[dict[str, Any]],  # Changed from ResponseInputParam
        tools: list[FunctionToolParam] | None = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
    ) -> AsyncGenerator[ResponsesAPIStreamingResponse, None]:
        """Perform a response request to the LLM."""


class LiteLLMClient(BaseLLMClient):
    async def completion(self, messages: list[Any], tools: list[ChatCompletionToolParam] | None = None, tool_choice: str = "auto") -> Any:  # noqa: ANN401
        """Perform a completion request to the Litellm API."""
        return await litellm.acompletion(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            api_version=self.api_version,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
        )

    async def responses(
        self,
        messages: list[dict[str, Any]],  # Changed from ResponseInputParam
        tools: list[FunctionToolParam] | None = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
    ) -> AsyncGenerator[ResponsesAPIStreamingResponse, None]:
        """Perform a response request to the Litellm API."""

        os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"

        return await litellm.aresponses(
            model=self.model,
            input=messages,  # type: ignore[arg-type]
            tools=tools,
            tool_choice=tool_choice,
            api_version=self.api_version,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
            store=False,
        )
