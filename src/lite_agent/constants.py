from typing import Literal


class CompletionMode:
    """Agent completion modes."""

    STOP: Literal["stop"] = "stop"  # Traditional completion until model decides to stop
    CALL: Literal["call"] = "call"  # Completion until specific tool is called


class ToolName:
    """System tool names."""

    TRANSFER_TO_AGENT = "transfer_to_agent"
    TRANSFER_TO_PARENT = "transfer_to_parent"
    WAIT_FOR_USER = "wait_for_user"


class StreamIncludes:
    """Default stream includes configuration."""

    DEFAULT_INCLUDES = (
        "completion_raw",
        "usage",
        "function_call",
        "function_call_output",
        "content_delta",
        "function_call_delta",
        "assistant_message",
    )
