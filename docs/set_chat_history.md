# Set Chat History Functionality

这个文档描述了 Runner 类新增的 `set_chat_history()` 功能。

## 功能概述

`set_chat_history()` 方法允许你：

1. **设置完整的聊天历史记录** - 一次性设置所有历史消息
2. **自动追踪当前 agent** - 根据历史记录中的函数调用自动确定应该激活哪个 agent
3. **支持多种消息格式** - 支持字典和 Pydantic 模型格式

## 基本用法

```python
from lite_agent.runner import Runner
from lite_agent.agent import Agent

# 创建 agents
parent = Agent(model="gpt-4.1", name="ParentAgent", instructions="主要 agent")
child = Agent(model="gpt-4.1", name="ChildAgent", instructions="子 agent")
parent.add_handoff(child)

runner = Runner(parent)

# 设置包含 agent 转换的聊天历史
chat_history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Let me transfer you to our specialist."},
    {
        "type": "function_call", 
        "function_call_id": "call_1", 
        "name": "transfer_to_agent", 
        "arguments": '{"name": "ChildAgent"}', 
        "content": ""
    },
    {
        "type": "function_call_output", 
        "call_id": "call_1", 
        "output": "Transferring to agent: ChildAgent"
    }
]

# 设置历史并自动追踪当前 agent
runner.set_chat_history(chat_history, root_agent=parent)
print(f"当前 agent: {runner.agent.name}")  # 输出: ChildAgent
```

## 支持的转换类型

### 1. 转换到指定 Agent

```python
{
    "type": "function_call",
    "function_call_id": "call_1",
    "name": "transfer_to_agent",
    "arguments": '{"name": "TargetAgentName"}',
    "content": ""
}
```

### 2. 转换回父 Agent

```python
{
    "type": "function_call",
    "function_call_id": "call_2",
    "name": "transfer_to_parent",
    "arguments": "{}",
    "content": ""
}
```

## 复杂转换示例

```python
# 复杂的转换序列：parent -> child1 -> parent -> child2
complex_history = [
    {"role": "user", "content": "I need help with weather and temperature"},
    
    # 转换到天气 agent
    {"type": "function_call", "function_call_id": "call_1", "name": "transfer_to_agent", 
     "arguments": '{"name": "WeatherAgent"}', "content": ""},
    {"type": "function_call_output", "call_id": "call_1", "output": "Transferring to agent: WeatherAgent"},
    {"role": "assistant", "content": "Weather information provided"},
    
    # 转换回父 agent
    {"type": "function_call", "function_call_id": "call_2", "name": "transfer_to_parent", 
     "arguments": "{}", "content": ""},
    {"type": "function_call_output", "call_id": "call_2", "output": "Transferring back to parent"},
    {"role": "assistant", "content": "Now let me get temperature info"},
    
    # 转换到温度 agent
    {"type": "function_call", "function_call_id": "call_3", "name": "transfer_to_agent", 
     "arguments": '{"name": "TemperatureAgent"}', "content": ""},
    {"type": "function_call_output", "call_id": "call_3", "output": "Transferring to agent: TemperatureAgent"}
]

runner.set_chat_history(complex_history, root_agent=parent)
print(f"最终 agent: {runner.agent.name}")  # 输出: TemperatureAgent
```

## 使用 Pydantic 模型

你也可以使用 Pydantic 模型对象：

```python
from lite_agent.types import AgentFunctionToolCallMessage, AgentFunctionCallOutput

transfer_message = AgentFunctionToolCallMessage(
    type="function_call",
    function_call_id="call_1",
    name="transfer_to_agent",
    arguments='{"name": "ChildAgent"}',
    content=""
)

output_message = AgentFunctionCallOutput(
    type="function_call_output",
    call_id="call_1",
    output="Transferring to agent: ChildAgent"
)

history = [
    {"role": "user", "content": "Hello"},
    transfer_message,
    output_message
]

runner.set_chat_history(history, root_agent=parent)
```

## 错误处理

方法能够优雅地处理各种错误情况：

- **无效的 agent 名称** - 继续使用当前 agent
- **格式错误的参数** - 跳过该转换，记录警告
- **没有父 agent 的 transfer_to_parent** - 继续使用当前 agent

## API 参考

### `Runner.set_chat_history(messages, root_agent=None)`

**参数:**
- `messages: list[FlexibleRunnerMessage]` - 要设置的消息列表
- `root_agent: Agent | None` - 根 agent，如果为 None 则使用 `self.agent`

**功能:**
- 清空当前消息历史
- 逐个添加新消息
- 追踪并设置正确的当前 agent
- 记录设置完成的信息

**返回:** 无

## 性能

该功能针对性能进行了优化：

- **处理速度**: ~500,000 消息/秒
- **内存效率**: 线性内存使用
- **复杂度**: O(n) 其中 n 是消息数量

## 测试

运行单元测试：

```bash
pytest tests/unit/test_set_chat_history.py -v
```

运行性能测试：

```bash
python tests/performance/test_set_chat_history_performance.py
```

## 使用场景

1. **会话恢复** - 从数据库或文件恢复之前的对话状态
2. **会话迁移** - 在不同的 Runner 实例之间迁移对话
3. **调试和测试** - 设置特定的对话状态进行测试
4. **会话分析** - 分析包含 agent 转换的历史对话
