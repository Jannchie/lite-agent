import abc
import os
from typing import Any, Literal, NotRequired, TypedDict

import litellm
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionToolParam
from openai.types.responses import FunctionToolParam, Response, ResponseStreamEvent
from pydantic import BaseModel

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ThinkingConfigDict(TypedDict):
    """Thinking configuration for reasoning models like Claude."""

    type: Literal["enabled"]  # 启用推理
    budget_tokens: NotRequired[int]  # 推理令牌预算，可选


class ReasoningEffortDict(TypedDict):
    """Reasoning effort configuration."""

    effort: ReasoningEffort


ThinkingConfig = ThinkingConfigDict | None

# 统一的推理配置类型
ReasoningConfig = (
    ReasoningEffort  # "minimal", "low", "medium", "high"
    | ReasoningEffortDict  # {"effort": "minimal"}
    | ThinkingConfigDict  # {"type": "enabled", "budget_tokens": 2048}
    | bool  # True/False 简单开关
    | None  # 不启用推理
)


class LLMConfig(BaseModel):
    """LLM generation parameters configuration."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    response_format: type[BaseModel] | dict[str, Any] | None = None


def parse_reasoning_config(reasoning: ReasoningConfig) -> tuple[ReasoningEffort | None, ThinkingConfig]:
    """
    解析统一的推理配置，返回 reasoning_effort 和 thinking_config。

    Args:
        reasoning: 统一的推理配置
            - ReasoningEffort: "minimal", "low", "medium", "high" -> reasoning_effort
            - ReasoningEffortDict: {"effort": "minimal"} -> reasoning_effort
            - ThinkingConfigDict: {"type": "enabled", "budget_tokens": 2048} -> thinking_config
            - bool: True -> "medium", False -> None
            - None: 不启用推理

    Returns:
        tuple: (reasoning_effort, thinking_config)
    """
    if reasoning is None:
        return None, None

    if isinstance(reasoning, str) and reasoning in ("minimal", "low", "medium", "high"):
        return reasoning, None  # type: ignore[return-value]

    if isinstance(reasoning, bool):
        return ("medium", None) if reasoning else (None, None)

    if isinstance(reasoning, dict):
        return _parse_dict_reasoning_config(reasoning)

    # 其他类型或无效格式，默认不启用
    return None, None


def _parse_dict_reasoning_config(reasoning: ReasoningEffortDict | ThinkingConfigDict | dict[str, Any]) -> tuple[ReasoningEffort | None, ThinkingConfig]:
    """解析字典格式的推理配置。"""
    # 检查是否为 {"effort": "value"} 格式 (ReasoningEffortDict)
    if "effort" in reasoning and len(reasoning) == 1:
        effort = reasoning["effort"]
        if isinstance(effort, str) and effort in ("minimal", "low", "medium", "high"):
            return effort, None  # type: ignore[return-value]

    # 检查是否为 ThinkingConfigDict 格式
    if "type" in reasoning and reasoning.get("type") == "enabled":
        # 验证 ThinkingConfigDict 的结构
        valid_keys = {"type", "budget_tokens"}
        if all(key in valid_keys for key in reasoning):
            return None, reasoning  # type: ignore[return-value]

    # 其他未知字典格式，仍尝试作为 thinking_config
    return None, reasoning  # type: ignore[return-value]


def _prepare_response_format(
    response_format: type[BaseModel] | dict[str, Any] | None,
) -> dict[str, Any] | type[BaseModel] | None:
    """Normalize response_format to the schema accepted by the OpenAI SDK."""
    if response_format is None:
        return None

    # 如果是 Pydantic 模型类
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        schema = response_format.model_json_schema()

        # 确保符合 OpenAI structured output 要求
        def make_schema_strict(schema_dict: Any) -> None:  # noqa: ANN401
            if isinstance(schema_dict, dict):
                # 对于对象类型，设置 additionalProperties: false 和确保 required 包含所有属性
                if schema_dict.get("type") == "object":
                    schema_dict["additionalProperties"] = False
                    if "properties" in schema_dict:
                        # 确保所有属性都在 required 数组中
                        all_properties = list(schema_dict["properties"].keys())
                        schema_dict["required"] = all_properties

                # 递归处理所有嵌套结构
                for value in schema_dict.values():
                    make_schema_strict(value)

            elif isinstance(schema_dict, list):
                for item in schema_dict:
                    make_schema_strict(item)

        make_schema_strict(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    # 如果已经是字典格式，直接返回
    if isinstance(response_format, dict):
        return response_format

    return None


class BaseLLMClient(abc.ABC):
    """Base class for LLM clients."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        reasoning: ReasoningConfig = None,
        llm_config: LLMConfig | None = None,
        **llm_params: Any,  # noqa: ANN401
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

        # 处理 LLM 生成参数
        if llm_config is not None:
            self.llm_config = llm_config
        else:
            # 从 **llm_params 创建配置
            self.llm_config = LLMConfig(**llm_params)

        # 处理推理配置
        self.reasoning_effort: ReasoningEffort | None
        self.thinking_config: ThinkingConfig
        self.reasoning_effort, self.thinking_config = parse_reasoning_config(reasoning)

    def _resolve_reasoning_params(
        self,
        reasoning: ReasoningConfig,
    ) -> tuple[ReasoningEffort | None, ThinkingConfig]:
        """Resolve reasoning configuration for a single request."""
        if reasoning is not None:
            return parse_reasoning_config(reasoning)
        return self.reasoning_effort, self.thinking_config

    @abc.abstractmethod
    async def completion(
        self,
        messages: list[Any],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: str = "auto",
        reasoning: ReasoningConfig = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Perform a completion request to the LLM."""

    @abc.abstractmethod
    async def responses(
        self,
        messages: list[dict[str, Any]],  # Changed from ResponseInputParam
        tools: list[FunctionToolParam] | None = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        reasoning: ReasoningConfig = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Perform a response request to the LLM."""


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        organization: str | None = None,
        reasoning: ReasoningConfig = None,
        llm_config: LLMConfig | None = None,
        **llm_params: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            reasoning=reasoning,
            llm_config=llm_config,
            **llm_params,
        )

        client_kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            client_kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            client_kwargs["base_url"] = self.api_base
        if organization is not None:
            client_kwargs["organization"] = organization

        self._client = AsyncOpenAI(**client_kwargs)

    async def completion(
        self,
        messages: list[Any],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: str = "auto",
        reasoning: ReasoningConfig = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """Perform a completion request using the OpenAI SDK."""

        final_reasoning_effort, _ = self._resolve_reasoning_params(reasoning)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        if tools is not None:
            params["tools"] = tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        if streaming:
            params["stream"] = True

        llm_config = self.llm_config
        if llm_config.temperature is not None:
            params["temperature"] = llm_config.temperature
        if llm_config.max_tokens is not None:
            params["max_tokens"] = llm_config.max_tokens
        if llm_config.top_p is not None:
            params["top_p"] = llm_config.top_p
        if llm_config.frequency_penalty is not None:
            params["frequency_penalty"] = llm_config.frequency_penalty
        if llm_config.presence_penalty is not None:
            params["presence_penalty"] = llm_config.presence_penalty
        if llm_config.stop is not None:
            params["stop"] = llm_config.stop

        final_response_format = response_format or llm_config.response_format
        prepared_response_format = _prepare_response_format(final_response_format)
        if prepared_response_format is not None:
            params["response_format"] = prepared_response_format

        if final_reasoning_effort is not None:
            params["reasoning_effort"] = final_reasoning_effort

        return await self._client.chat.completions.create(**params)

    async def responses(
        self,
        messages: list[dict[str, Any]],
        tools: list[FunctionToolParam] | None = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        reasoning: ReasoningConfig = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> Response | AsyncStream[ResponseStreamEvent]:
        """Perform a Responses API request using the OpenAI SDK."""

        final_reasoning_effort, _ = self._resolve_reasoning_params(reasoning)

        params: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "store": False,
            **kwargs,
        }

        if tools is not None:
            params["tools"] = tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        if streaming:
            params["stream"] = True

        llm_config = self.llm_config
        if llm_config.temperature is not None:
            params["temperature"] = llm_config.temperature
        if llm_config.max_tokens is not None:
            params["max_tokens"] = llm_config.max_tokens
        if llm_config.top_p is not None:
            params["top_p"] = llm_config.top_p
        if llm_config.frequency_penalty is not None:
            params["frequency_penalty"] = llm_config.frequency_penalty
        if llm_config.presence_penalty is not None:
            params["presence_penalty"] = llm_config.presence_penalty
        if llm_config.stop is not None:
            params["stop"] = llm_config.stop

        final_response_format = response_format or llm_config.response_format
        prepared_response_format = _prepare_response_format(final_response_format)
        if prepared_response_format is not None:
            params["response_format"] = prepared_response_format

        if final_reasoning_effort is not None:
            params["reasoning"] = {"effort": final_reasoning_effort}

        return await self._client.responses.create(**params)


class LiteLLMClient(BaseLLMClient):
    """LiteLLM-based client that proxies requests to provider-specific backends."""

    async def completion(
        self,
        messages: list[Any],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: str = "auto",
        reasoning: ReasoningConfig = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Perform a completion request using LiteLLM."""

        final_reasoning_effort, final_thinking_config = self._resolve_reasoning_params(reasoning)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_version": self.api_version,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "stream": streaming,
            **kwargs,
        }
        if tools is not None:
            params["tools"] = tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        if streaming:
            params["stream"] = True
            params["stream_options"] = {**params.get("stream_options", {}), "include_usage": True}

        llm_config = self.llm_config
        if llm_config.temperature is not None:
            params["temperature"] = llm_config.temperature
        if llm_config.max_tokens is not None:
            params["max_tokens"] = llm_config.max_tokens
        if llm_config.top_p is not None:
            params["top_p"] = llm_config.top_p
        if llm_config.frequency_penalty is not None:
            params["frequency_penalty"] = llm_config.frequency_penalty
        if llm_config.presence_penalty is not None:
            params["presence_penalty"] = llm_config.presence_penalty
        if llm_config.stop is not None:
            params["stop"] = llm_config.stop

        final_response_format = response_format or llm_config.response_format
        prepared_response_format = _prepare_response_format(final_response_format)
        if prepared_response_format is not None:
            params["response_format"] = prepared_response_format

        if final_reasoning_effort is not None:
            params["reasoning_effort"] = final_reasoning_effort
        if final_thinking_config is not None:
            params["thinking"] = final_thinking_config

        return await litellm.acompletion(**params)

    async def responses(
        self,
        messages: list[dict[str, Any]],  # Changed from ResponseInputParam
        tools: list[FunctionToolParam] | None = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        reasoning: ReasoningConfig = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Perform a Responses API request using LiteLLM."""

        os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"

        final_reasoning_effort, final_thinking_config = self._resolve_reasoning_params(reasoning)

        params: dict[str, Any] = {
            "model": self.model,
            "input": messages,  # type: ignore[arg-type]
            "api_version": self.api_version,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "stream": streaming,
            "store": False,
            **kwargs,
        }
        if tools is not None:
            params["tools"] = tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        if streaming:
            params["stream"] = True
            params["stream_options"] = {**params.get("stream_options", {}), "include_usage": True}

        llm_config = self.llm_config
        if llm_config.temperature is not None:
            params["temperature"] = llm_config.temperature
        if llm_config.max_tokens is not None:
            params["max_tokens"] = llm_config.max_tokens
        if llm_config.top_p is not None:
            params["top_p"] = llm_config.top_p
        if llm_config.frequency_penalty is not None:
            params["frequency_penalty"] = llm_config.frequency_penalty
        if llm_config.presence_penalty is not None:
            params["presence_penalty"] = llm_config.presence_penalty
        if llm_config.stop is not None:
            params["stop"] = llm_config.stop

        final_response_format = response_format or llm_config.response_format
        prepared_response_format = _prepare_response_format(final_response_format)
        if prepared_response_format is not None:
            params["response_format"] = prepared_response_format

        if final_reasoning_effort is not None:
            params["reasoning"] = {"effort": final_reasoning_effort}
        if final_thinking_config is not None:
            params["thinking"] = final_thinking_config

        return await litellm.aresponses(**params)  # type: ignore[no-any-return]
