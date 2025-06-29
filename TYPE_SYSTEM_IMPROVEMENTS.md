# Runner 类型系统改进

## 概述

我对 `lite-agent` 项目的 `Runner` 类进行了类型系统改进，使 `run` 方法的 `user_input` 参数能够支持更多类型，并提供更好的类型提示。

## 主要改进

### 1. 新增 TypedDict 定义

在 `src/lite_agent/types/messages.py` 中添加了以下 TypedDict 类型：

```python
class UserMessageDict(TypedDict):
    role: Literal["user"]
    content: str

class AssistantMessageDict(TypedDict):
    role: Literal["assistant"]
    content: str

class SystemMessageDict(TypedDict):
    role: Literal["system"]
    content: str

class FunctionCallDict(TypedDict):
    type: Literal["function_call"]
    function_call_id: str
    name: str
    arguments: str
    content: str

class FunctionCallOutputDict(TypedDict):
    type: Literal["function_call_output"]
    call_id: str
    output: str
```

### 2. 增强的类型联合

创建了新的类型联合来支持更灵活的输入：

```python
# 支持 BaseModel、TypedDict 和普通字典的联合类型
MessageDict = UserMessageDict | AssistantMessageDict | SystemMessageDict | FunctionCallDict | FunctionCallOutputDict

# 灵活的消息类型，支持所有格式
FlexibleRunnerMessage = RunnerMessage | MessageDict | dict[str, Any]

# 序列类型
RunnerMessages = Sequence[FlexibleRunnerMessage]

# 用户输入的完整类型定义
UserInput = str | FlexibleRunnerMessage | RunnerMessages
```

### 3. 更新 Runner.run() 方法

`Runner` 类的 `run` 方法现在接受 `UserInput` 类型：

```python
def run(
    self,
    user_input: UserInput,  # 新的灵活类型
    max_steps: int = 20,
    includes: Sequence[AgentChunkType] | None = None,
    context: "Any | None" = None,
    record_to: PathLike | str | None = None,
) -> AsyncGenerator[AgentChunk, None]:
```

### 4. 改进的 append_message 方法

更新了 `append_message` 方法来处理新的类型：

```python
def append_message(self, message: FlexibleRunnerMessage) -> None:
```

## 使用方式

### 1. 字符串输入（最简单）
```python
await runner.run("Hello, how are you?")
```

### 2. BaseModel 实例（Pydantic 模型）
```python
user_message = AgentUserMessage(
    role="user",
    content="This is a BaseModel message"
)
await runner.run(user_message)
```

### 3. TypedDict（获得优秀的类型提示）
```python
user_msg: UserMessageDict = {
    "role": "user",  # IDE 提示：只能是 "user"
    "content": "Hello"  # IDE 知道这是必需字段
}
await runner.run(user_msg)
```

### 4. 普通字典（向后兼容）
```python
user_msg = {
    "role": "user",
    "content": "This works too"
}
await runner.run(user_msg)
```

### 5. 消息序列
```python
conversation: list[UserMessageDict | AssistantMessageDict] = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]
await runner.run(conversation)
```

## 优势

1. **更好的类型提示**：TypedDict 提供结构化类型提示，IDE 能够准确提示可用字段和值
2. **类型安全**：编译时类型检查可以捕获错误
3. **向后兼容**：继续支持原有的字典格式
4. **灵活性**：支持 BaseModel、TypedDict 和普通字典
5. **开发体验**：更好的自动完成和错误检测

## 示例文件

参见 `examples/type_system_example.py` 了解完整的使用示例。

## 影响范围

- `src/lite_agent/types/messages.py`：新增 TypedDict 定义
- `src/lite_agent/types/__init__.py`：导出新类型
- `src/lite_agent/runner.py`：更新 `run` 和 `append_message` 方法
- `examples/type_system_example.py`：新增示例文件

这些改进保持了完全的向后兼容性，同时为开发者提供了更好的类型安全性和开发体验。
