import time
from collections.abc import AsyncGenerator, Callable, Sequence
from pathlib import Path
from typing import Any, Optional

from funcall import Funcall
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from lite_agent.client import BaseLLMClient, LiteLLMClient
from lite_agent.constants import CompletionMode, ToolName
from lite_agent.loggers import logger
from lite_agent.response_handlers import CompletionResponseHandler, ResponsesAPIHandler
from lite_agent.types import (
    AgentChunk,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunnerMessages,
    ToolCall,
    system_message_to_llm_dict,
)
from lite_agent.types.messages import NewSystemMessage
from lite_agent.utils.message_converter import MessageFormatConverter, ResponsesFormatConverter

TEMPLATES_DIR = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)

HANDOFFS_SOURCE_INSTRUCTIONS_TEMPLATE = jinja_env.get_template("handoffs_source_instructions.xml.j2")
HANDOFFS_TARGET_INSTRUCTIONS_TEMPLATE = jinja_env.get_template("handoffs_target_instructions.xml.j2")
WAIT_FOR_USER_INSTRUCTIONS_TEMPLATE = jinja_env.get_template("wait_for_user_instructions.xml.j2")


class Agent:
    def __init__(
        self,
        *,
        model: str | BaseLLMClient,
        name: str,
        instructions: str,
        tools: list[Callable] | None = None,
        handoffs: list["Agent"] | None = None,
        message_transfer: Callable[[RunnerMessages], RunnerMessages] | None = None,
        completion_condition: str = "stop",
        stop_before_tools: list[str] | list[Callable] | None = None,
        termination_tools: list[str] | list[Callable] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        # Convert stop_before_functions to function names
        if stop_before_tools:
            self.stop_before_functions = set()
            for func in stop_before_tools:
                if isinstance(func, str):
                    self.stop_before_functions.add(func)
                elif callable(func):
                    self.stop_before_functions.add(func.__name__)
                else:
                    msg = f"stop_before_functions must contain strings or callables, got {type(func)}"
                    raise TypeError(msg)
        else:
            self.stop_before_functions = set()

        # Convert termination_tools to function names
        if termination_tools:
            self.termination_tools = set()
            for func in termination_tools:
                if isinstance(func, str):
                    self.termination_tools.add(func)
                elif callable(func):
                    self.termination_tools.add(func.__name__)
                else:
                    msg = f"termination_tools must contain strings or callables, got {type(func)}"
                    raise TypeError(msg)
        else:
            self.termination_tools = set()

        if isinstance(model, BaseLLMClient):
            # If model is a BaseLLMClient instance, use it directly
            self.client = model
        else:
            # Otherwise, create a LiteLLMClient instance
            self.client = LiteLLMClient(
                model=model,
            )
        self.completion_condition = completion_condition
        self.handoffs = handoffs if handoffs else []
        self._parent: Agent | None = None
        self.message_transfer = message_transfer
        self.response_format = response_format
        # Initialize Funcall with regular tools
        self.fc = Funcall(tools)

        # Add wait_for_user tool if completion condition is "call"
        if completion_condition == CompletionMode.CALL:
            self._add_wait_for_user_tool()

        # Set parent for handoff agents
        if handoffs:
            for handoff_agent in handoffs:
                handoff_agent.parent = self
            self._add_transfer_tools(handoffs)

        # Add transfer_to_parent tool if this agent has a parent (for cases where parent is set externally)
        if self.parent is not None:
            self.add_transfer_to_parent_tool()

    @property
    def parent(self) -> Optional["Agent"]:
        return self._parent

    @parent.setter
    def parent(self, value: Optional["Agent"]) -> None:
        self._parent = value
        if value is not None:
            self.add_transfer_to_parent_tool()

    def _add_transfer_tools(self, handoffs: list["Agent"]) -> None:
        """Add transfer function for handoff agents using dynamic tools.

        Creates a single 'transfer_to_agent' function that accepts a 'name' parameter
        to specify which agent to transfer the conversation to.

        Args:
            handoffs: List of Agent objects that can be transferred to
        """
        # Collect all agent names for validation
        agent_names = [agent.name for agent in handoffs]

        def transfer_handler(name: str) -> str:
            """Handler for transfer_to_agent function."""
            if name in agent_names:
                return f"Transferring to agent: {name}"

            available_agents = ", ".join(agent_names)
            return f"Agent '{name}' not found. Available agents: {available_agents}"

        # Add single dynamic tool for all transfers
        self.fc.add_dynamic_tool(
            name=ToolName.TRANSFER_TO_AGENT,
            description="Transfer conversation to another agent.",
            parameters={
                "name": {
                    "type": "string",
                    "description": "The name of the agent to transfer to",
                    "enum": agent_names,
                },
            },
            required=["name"],
            handler=transfer_handler,
        )

    def add_transfer_to_parent_tool(self) -> None:
        """Add transfer_to_parent function for agents that have a parent.

        This tool allows the agent to transfer back to its parent when:
        - The current task is completed
        - The agent cannot solve the current problem
        - Escalation to a higher level is needed
        """

        def transfer_to_parent_handler() -> str:
            """Handler for transfer_to_parent function."""
            if self.parent:
                return f"Transferring back to parent agent: {self.parent.name}"
            return "No parent agent found"

        # Add dynamic tool for parent transfer
        self.fc.add_dynamic_tool(
            name=ToolName.TRANSFER_TO_PARENT,
            description="Transfer conversation back to parent agent when current task is completed or cannot be solved by current agent",
            parameters={},
            required=[],
            handler=transfer_to_parent_handler,
        )

    def add_handoff(self, agent: "Agent") -> None:
        """Add a handoff agent after initialization.

        This method allows adding handoff agents dynamically after the agent
        has been constructed. It properly sets up parent-child relationships
        and updates the transfer tools.

        Args:
            agent: The agent to add as a handoff target
        """
        # Add to handoffs list if not already present
        if agent not in self.handoffs:
            self.handoffs.append(agent)

            # Set parent relationship
            agent.parent = self

            # Add transfer_to_parent tool to the handoff agent
            agent.add_transfer_to_parent_tool()

            # Remove existing transfer tool if it exists and recreate with all agents
            try:
                # Try to remove the existing transfer tool
                if hasattr(self.fc, "remove_dynamic_tool"):
                    self.fc.remove_dynamic_tool(ToolName.TRANSFER_TO_AGENT)
            except Exception as e:
                # If removal fails, log and continue anyway
                logger.debug(f"Failed to remove existing transfer tool: {e}")

            # Regenerate transfer tools to include the new agent
            self._add_transfer_tools(self.handoffs)

    def _build_instructions(self) -> str:
        """Build complete instructions with templates."""
        instructions = self.instructions
        if self.handoffs:
            instructions = HANDOFFS_SOURCE_INSTRUCTIONS_TEMPLATE.render(extra_instructions=None) + "\n\n" + instructions
        if self.parent:
            instructions = HANDOFFS_TARGET_INSTRUCTIONS_TEMPLATE.render(extra_instructions=None) + "\n\n" + instructions
        if self.completion_condition == "call":
            instructions = WAIT_FOR_USER_INSTRUCTIONS_TEMPLATE.render(extra_instructions=None) + "\n\n" + instructions
        return instructions

    def prepare_completion_messages(self, messages: RunnerMessages) -> list[dict]:
        """Prepare messages for completions API (with conversion)."""
        converted_messages = MessageFormatConverter.to_completion_format(messages)
        instructions = self._build_instructions()
        return [
            system_message_to_llm_dict(
                NewSystemMessage(
                    content=f"You are {self.name}. {instructions}",
                ),
            ),
            *converted_messages,
        ]

    def prepare_responses_messages(self, messages: RunnerMessages) -> list[dict[str, Any]]:
        """Prepare messages for responses API (no conversion, just add system message if needed)."""
        instructions = self._build_instructions()
        converted_messages = ResponsesFormatConverter.to_responses_format(messages)
        return [
            {
                "role": "system",
                "content": f"You are {self.name}. {instructions}",
            },
            *converted_messages,
        ]

    async def completion(
        self,
        messages: RunnerMessages,
        record_to_file: Path | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
    ) -> AsyncGenerator[AgentChunk, None]:
        # Apply message transfer callback if provided
        processed_messages = messages
        if self.message_transfer:
            logger.debug(f"Applying message transfer callback for agent {self.name}")
            processed_messages = self.message_transfer(messages)

        # For completions API, use prepare_completion_messages
        self.message_histories = self.prepare_completion_messages(processed_messages)

        tools = self.fc.get_tools(target="completion")
        # 优先使用方法参数，然后是实例属性
        final_response_format = response_format or self.response_format
        resp = await self.client.completion(
            messages=self.message_histories,
            tools=tools,
            tool_choice="auto",  # TODO: make this configurable
            response_format=final_response_format,
            streaming=streaming,
        )

        # Use response handler for unified processing
        handler = CompletionResponseHandler()
        return handler.handle(resp, streaming=streaming, record_to=record_to_file)

    async def responses(
        self,
        messages: RunnerMessages,
        record_to_file: Path | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        *,
        streaming: bool = True,
    ) -> AsyncGenerator[AgentChunk, None]:
        # Apply message transfer callback if provided
        processed_messages = messages
        if self.message_transfer:
            logger.debug(f"Applying message transfer callback for agent {self.name}")
            processed_messages = self.message_transfer(messages)

        # For responses API, use prepare_responses_messages (no conversion)
        self.message_histories = self.prepare_responses_messages(processed_messages)
        tools = self.fc.get_tools()
        # 优先使用方法参数，然后是实例属性
        final_response_format = response_format or self.response_format
        resp = await self.client.responses(
            messages=self.message_histories,
            tools=tools,
            tool_choice="auto",  # TODO: make this configurable
            response_format=final_response_format,
            streaming=streaming,
        )
        # Use response handler for unified processing
        handler = ResponsesAPIHandler()
        return handler.handle(resp, streaming=streaming, record_to=record_to_file)

    async def list_require_confirm_tools(self, tool_calls: Sequence[ToolCall] | None) -> Sequence[ToolCall]:
        if not tool_calls:
            return []
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Check if function is in dynamic stop_before_functions list
            if function_name in self.stop_before_functions:
                logger.debug('Tool call "%s" requires confirmation (stop_before_functions)', tool_call.id)
                results.append(tool_call)
                continue

            # Check decorator-based require_confirmation
            tool_func = self.fc.function_registry.get(function_name)
            if not tool_func:
                logger.warning("Tool function %s not found in registry", function_name)
                continue
            tool_meta = self.fc.get_tool_meta(function_name)
            if tool_meta["require_confirm"]:
                logger.debug('Tool call "%s" requires confirmation (decorator)', tool_call.id)
                results.append(tool_call)
        return results

    async def handle_tool_calls(self, tool_calls: Sequence[ToolCall] | None, context: Any | None = None) -> AsyncGenerator[FunctionCallEvent | FunctionCallOutputEvent, None]:  # noqa: ANN401
        if not tool_calls:
            return
        if tool_calls:
            for tool_call in tool_calls:
                tool_func = self.fc.function_registry.get(tool_call.function.name)
                if not tool_func:
                    logger.warning("Tool function %s not found in registry", tool_call.function.name)
                    continue

            for tool_call in tool_calls:
                yield FunctionCallEvent(
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments or "",
                )
                start_time = time.time()
                try:
                    content = await self.fc.call_function_async(tool_call.function.name, tool_call.function.arguments or "", context)
                    end_time = time.time()
                    execution_time_ms = int((end_time - start_time) * 1000)
                    yield FunctionCallOutputEvent(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        content=str(content),
                        execution_time_ms=execution_time_ms,
                    )
                except Exception as e:
                    logger.exception("Tool call %s failed", tool_call.id)
                    end_time = time.time()
                    execution_time_ms = int((end_time - start_time) * 1000)
                    yield FunctionCallOutputEvent(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        content=str(e),
                        execution_time_ms=execution_time_ms,
                    )

    def set_message_transfer(self, message_transfer: Callable[[RunnerMessages], RunnerMessages] | None) -> None:
        """Set or update the message transfer callback function.

        Args:
            message_transfer: A callback function that takes RunnerMessages as input
                             and returns RunnerMessages as output. This function will be
                             called before making API calls to allow preprocessing of messages.
        """
        self.message_transfer = message_transfer

    def _add_wait_for_user_tool(self) -> None:
        """Add wait_for_user tool for agents with completion_condition='call'.

        This tool allows the agent to signal when it has completed its task.
        """

        def wait_for_user_handler() -> str:
            """Handler for wait_for_user function."""
            return "Waiting for user input."

        # Add dynamic tool for task completion
        self.fc.add_dynamic_tool(
            name=ToolName.WAIT_FOR_USER,
            description="Call this function when you have completed your assigned task or need more information from the user.",
            parameters={},
            required=[],
            handler=wait_for_user_handler,
        )

    def set_stop_before_functions(self, functions: list[str] | list[Callable]) -> None:
        """Set the list of functions that require confirmation before execution.

        Args:
            functions: List of function names (str) or callable objects
        """
        self.stop_before_functions = set()
        for func in functions:
            if isinstance(func, str):
                self.stop_before_functions.add(func)
            elif callable(func):
                self.stop_before_functions.add(func.__name__)
            else:
                msg = f"stop_before_functions must contain strings or callables, got {type(func)}"
                raise TypeError(msg)
        logger.debug(f"Set stop_before_functions to: {self.stop_before_functions}")

    def add_stop_before_function(self, function: str | Callable) -> None:
        """Add a function to the stop_before_functions list.

        Args:
            function: Function name (str) or callable object to add
        """
        if isinstance(function, str):
            function_name = function
        elif callable(function):
            function_name = function.__name__
        else:
            msg = f"function must be a string or callable, got {type(function)}"
            raise TypeError(msg)

        self.stop_before_functions.add(function_name)
        logger.debug(f"Added '{function_name}' to stop_before_functions")

    def remove_stop_before_function(self, function: str | Callable) -> None:
        """Remove a function from the stop_before_functions list.

        Args:
            function: Function name (str) or callable object to remove
        """
        if isinstance(function, str):
            function_name = function
        elif callable(function):
            function_name = function.__name__
        else:
            msg = f"function must be a string or callable, got {type(function)}"
            raise TypeError(msg)

        self.stop_before_functions.discard(function_name)
        logger.debug(f"Removed '{function_name}' from stop_before_functions")

    def clear_stop_before_functions(self) -> None:
        """Clear all function names from the stop_before_functions list."""
        self.stop_before_functions.clear()
        logger.debug("Cleared all stop_before_functions")

    def get_stop_before_functions(self) -> set[str]:
        """Get the current set of function names that require confirmation.

        Returns:
            Set of function names
        """
        return self.stop_before_functions.copy()

    def set_response_format(self, response_format: type[BaseModel] | dict[str, Any] | None) -> None:
        """Set the response format for structured output.

        Args:
            response_format: Pydantic model class, dict format, or None to disable
        """
        self.response_format = response_format

    def get_response_format(self) -> type[BaseModel] | dict[str, Any] | None:
        """Get the current response format.

        Returns:
            Current response format setting
        """
        return self.response_format
