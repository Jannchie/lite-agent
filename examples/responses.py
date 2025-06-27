import litellm
from funcall import Funcall
from openai.types.responses.response_input_param import ResponseInputParam


def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    return f"The temperature in {city} is 25°C."


def get_whether(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny with a few clouds."


fc = Funcall([get_temperature, get_whether])

messages: ResponseInputParam = [
    {
        "role": "system",
        "content": "You are a helpful weather assistant. Before using tools, briefly explain what you are going to do. Provide friendly and informative responses.",
    },
    {
        "role": "user",
        "content": "What is the weather in New York?",
    },
    {
        "arguments": '{"city":"New York"}',
        "call_id": "call_V23uiXgXRlv9pRoW4qAgflKF",
        "name": "get_whether",
        "type": "function_call",
    },
    {
        "call_id": "call_V23uiXgXRlv9pRoW4qAgflKF",
        "output": '{"result": 42, "msg": "done"}',
        "type": "function_call_output",
    },
]

resp = litellm.responses(model="gpt-4.1-nano", input=messages, tools=fc.get_tools(), tool_choice="auto", stream=False)

print(resp)
