"""
测试 response_event_processor 模块的功能
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from litellm.types.llms.openai import (
    ContentPartAddedEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIStreamEvents,
)

from lite_agent.processors.response_event_processor import ResponseEventProcessor
from lite_agent.types import (
    AssistantMessageEvent,
    ContentDeltaEvent,
    FunctionCallEvent,
    TimingEvent,
    UsageEvent,
)


class TestResponseEventProcessor:
    """测试ResponseEventProcessor类"""

    def test_init(self):
        """测试初始化"""
        processor = ResponseEventProcessor()
        assert processor._messages == []
        assert processor._start_time is None
        assert processor._first_output_time is None
        assert processor._output_complete_time is None
        assert processor._usage_time is None
        assert processor._usage_data == {}

    @pytest.mark.asyncio
    async def test_process_chunk_empty_events(self):
        """测试处理空事件列表"""
        processor = ResponseEventProcessor()

        # 创建一个模拟的chunk
        mock_chunk = Mock()
        mock_chunk.model_dump_json.return_value = '{"test": "data"}'
        mock_chunk.type = "test_type"

        chunks = []
        async for chunk in processor.process_chunk(mock_chunk):
            chunks.append(chunk)

        # 应该至少包含一个ResponseRawEvent
        assert len(chunks) >= 1
        assert chunks[0].type == "response_raw"

    @pytest.mark.asyncio
    async def test_process_chunk_yields_multiple_events(self):
        """测试处理返回多个事件的chunk"""
        processor = ResponseEventProcessor()

        # 创建一个返回多个事件的 mock chunk
        mock_chunk = Mock()
        mock_chunk.model_dump_json.return_value = '{"test": "data"}'
        mock_chunk.type = "test_type"

        # 模拟 handle_event返回多个事件
        mock_events = [Mock(), Mock()]
        processor.handle_event = Mock(return_value=mock_events)

        chunks = []
        async for chunk in processor.process_chunk(mock_chunk):
            chunks.append(chunk)

        # 应该有 1 个 ResponseRawEvent + 2 个模拟事件
        assert len(chunks) == 3
        assert chunks[0].type == "response_raw"
        # 后两个是来自 handle_event 的返回值
        assert chunks[1] == mock_events[0]
        assert chunks[2] == mock_events[1]

    @pytest.mark.asyncio
    async def test_process_chunk_timing_events(self):
        """测试处理时间事件"""
        processor = ResponseEventProcessor()

        # 模拟不同的事件类型
        mock_chunk = Mock()
        mock_chunk.model_dump_json.return_value = '{"test": "data"}'
        mock_chunk.type = "test_type"

        chunks = []
        async for chunk in processor.process_chunk(mock_chunk):
            chunks.append(chunk)

        # 测试时间记录
        assert processor._start_time is not None

    @pytest.mark.asyncio
    async def test_process_chunk_with_record_file(self):
        """测试带记录文件的处理"""
        processor = ResponseEventProcessor()

        mock_record_file = AsyncMock()
        mock_chunk = Mock()
        mock_chunk.model_dump_json.return_value = '{"test": "data"}'
        mock_chunk.type = "test_type"

        chunks = []
        async for chunk in processor.process_chunk(mock_chunk, record_file=mock_record_file):
            chunks.append(chunk)

        # 验证记录文件被调用
        mock_record_file.write.assert_called_once_with('{"test": "data"}\n')
        mock_record_file.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_multiple_chunks(self):
        """测试处理多个chunks"""
        processor = ResponseEventProcessor()

        # 创建多个模拟chunks
        chunks_to_process = []
        for i in range(3):
            mock_chunk = Mock()
            mock_chunk.model_dump_json.return_value = f'{{"test": "data{i}"}}'
            mock_chunk.type = f"test_type_{i}"
            chunks_to_process.append(mock_chunk)

        # 处理所有chunks
        all_output_chunks = []
        for chunk in chunks_to_process:
            async for output_chunk in processor.process_chunk(chunk):
                all_output_chunks.append(output_chunk)

        # 验证处理过程
        assert processor._start_time is not None
        assert len(all_output_chunks) >= 3  # 至少每个chunk产生一个ResponseRawEvent

    @pytest.mark.asyncio
    async def test_process_chunk_error_handling(self):
        """测试错误处理"""
        processor = ResponseEventProcessor()

        # 创建一个会引发异常的mock chunk
        mock_chunk = Mock()
        mock_chunk.model_dump_json.side_effect = ValueError("Test error")

        chunks = []
        try:
            async for chunk in processor.process_chunk(mock_chunk):
                chunks.append(chunk)
        except (ValueError, AttributeError):
            # 期望会有错误，这是正常的
            pass

        # 即使出错，start_time也应该被设置
        assert processor._start_time is not None

    def test_processor_state_management(self):
        """测试处理器状态管理"""
        processor = ResponseEventProcessor()

        # 测试初始状态
        assert len(processor._messages) == 0
        assert processor._start_time is None

        # 手动设置一些状态来测试
        test_time = datetime.now(timezone.utc)
        processor._start_time = test_time
        processor._usage_data = {"input_tokens": 10, "output_tokens": 20}

        assert processor._start_time == test_time
        assert processor._usage_data["input_tokens"] == 10
        assert processor._usage_data["output_tokens"] == 20

    @pytest.mark.asyncio
    async def test_chunk_generation_types(self):
        """测试生成的chunk类型"""
        processor = ResponseEventProcessor()

        mock_chunk = Mock()
        mock_chunk.model_dump_json.return_value = '{"test": "data"}'
        mock_chunk.type = "test_type"

        chunks = []
        async for chunk in processor.process_chunk(mock_chunk):
            chunks.append(chunk)
            # 验证chunk是AgentChunk类型
            assert hasattr(chunk, "type")

    def test_internal_data_structures(self):
        """测试内部数据结构"""
        processor = ResponseEventProcessor()

        # 测试默认的内部数据结构
        assert isinstance(processor._messages, list)
        assert isinstance(processor._usage_data, dict)

        # 测试数据结构的修改
        processor._messages.append({"test": "message"})
        processor._usage_data["test_key"] = "test_value"

        assert len(processor._messages) == 1
        assert processor._usage_data["test_key"] == "test_value"

    @pytest.mark.asyncio
    async def test_async_generator_behavior(self):
        """测试异步生成器行为"""
        processor = ResponseEventProcessor()

        mock_chunk = Mock()
        mock_chunk.model_dump_json.return_value = '{"test": "data"}'
        mock_chunk.type = "test_type"

        # 验证返回值是异步生成器
        result = processor.process_chunk(mock_chunk)
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")

        # 验证可以迭代
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # 应该至少有一个ResponseRawEvent
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_handle_event_ignored_types(self):
        """测试被忽略的事件类型"""
        processor = ResponseEventProcessor()

        # 测试被忽略的事件类型
        ignored_events = [
            Mock(type=ResponsesAPIStreamEvents.RESPONSE_CREATED),
            Mock(type=ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS),
            Mock(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE),
            Mock(type=ResponsesAPIStreamEvents.CONTENT_PART_DONE),
        ]

        for event in ignored_events:
            result = processor.handle_event(event)
            assert result == []

    def test_handle_output_item_added_event(self):
        """测试 OutputItemAddedEvent 处理"""
        processor = ResponseEventProcessor()

        # 创建 OutputItemAddedEvent
        mock_item = {"type": "message", "content": []}
        event = OutputItemAddedEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            output_index=0,
            item=mock_item,
        )

        result = processor.handle_event(event)

        assert result == []
        assert len(processor._messages) == 1
        assert processor._messages[0] == mock_item

    def test_handle_content_part_added_event(self):
        """测试 ContentPartAddedEvent 处理"""
        processor = ResponseEventProcessor()

        # 先添加一个消息
        processor._messages.append({"content": []})

        # 创建 ContentPartAddedEvent
        mock_part = {"type": "text", "text": "Hello"}
        event = ContentPartAddedEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
            item_id="item_123",
            output_index=0,
            content_index=0,
            part=mock_part,
        )

        result = processor.handle_event(event)

        assert result == []
        assert len(processor._messages[0]["content"]) == 1
        assert processor._messages[0]["content"][0] == mock_part

    def test_handle_output_text_delta_event(self):
        """测试 OutputTextDeltaEvent 处理"""
        processor = ResponseEventProcessor()

        # 先添加一个带文本内容的消息
        processor._messages.append(
            {
                "content": [{"text": "Hello", "type": "text"}],
            },
        )

        # 创建 OutputTextDeltaEvent
        event = OutputTextDeltaEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            item_id="item_123",
            output_index=0,
            content_index=0,
            delta=" World",
        )

        result = processor.handle_event(event)

        assert len(result) == 1
        assert isinstance(result[0], ContentDeltaEvent)
        assert result[0].delta == " World"
        assert processor._messages[0]["content"][0]["text"] == "Hello World"
        assert processor._first_output_time is not None

    def test_handle_output_item_done_event_function_call(self):
        """测试 OutputItemDoneEvent 函数调用处理"""
        processor = ResponseEventProcessor()

        # 创建函数调用的 OutputItemDoneEvent
        mock_item = {
            "type": "function_call",
            "call_id": "call_123",
            "name": "test_function",
            "arguments": '{"param": "value"}',
        }
        event = OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            output_index=0,
            item=mock_item,
        )

        result = processor.handle_event(event)

        assert len(result) == 1
        assert isinstance(result[0], FunctionCallEvent)
        assert result[0].call_id == "call_123"
        assert result[0].name == "test_function"
        assert result[0].arguments == '{"param": "value"}'

    def test_handle_output_item_done_event_message(self):
        """测试 OutputItemDoneEvent 消息处理"""
        processor = ResponseEventProcessor()
        # 设置时间以便测试时间计算
        processor._start_time = datetime.now(timezone.utc)
        processor._first_output_time = datetime.now(timezone.utc)

        # 创建消息的 OutputItemDoneEvent
        mock_item = {
            "type": "message",
            "content": [{"text": "Hello", "type": "text"}],
        }
        event = OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            output_index=0,
            item=mock_item,
        )

        result = processor.handle_event(event)

        assert len(result) == 1
        assert isinstance(result[0], AssistantMessageEvent)
        assert processor._output_complete_time is not None

        # 检查消息元数据
        message = result[0].message
        assert message.meta is not None
        assert message.meta.latency_ms is not None
        assert message.meta.output_time_ms is not None

    def test_handle_function_call_arguments_delta_event(self):
        """测试 FunctionCallArgumentsDeltaEvent 处理"""
        processor = ResponseEventProcessor()

        # 先添加一个函数调用消息
        processor._messages.append(
            {
                "type": "function_call",
                "arguments": "{",
            },
        )

        # 使用 Mock 来模拟 FunctionCallArgumentsDeltaEvent
        event = Mock(spec=FunctionCallArgumentsDeltaEvent)
        event.delta = '"param":'
        # Mock 必须有 type 属性才能通过初始检查
        event.type = "other"

        result = processor.handle_event(event)

        assert result == []
        assert processor._messages[0]["arguments"] == '{"param":'

    def test_handle_function_call_arguments_delta_event_no_arguments(self):
        """测试 FunctionCallArgumentsDeltaEvent 处理（无 arguments 字段）"""
        processor = ResponseEventProcessor()

        # 先添加一个没有 arguments 字段的函数调用消息
        processor._messages.append(
            {
                "type": "function_call",
            },
        )

        # 使用 Mock 来模拟 FunctionCallArgumentsDeltaEvent
        event = Mock(spec=FunctionCallArgumentsDeltaEvent)
        event.delta = '{"param": "value"}'
        # Mock 必须有 type 属性才能通过初始检查
        event.type = "other"

        result = processor.handle_event(event)

        assert result == []
        # 应该初始化为空字符串，然后添加 delta
        assert processor._messages[0]["arguments"] == '{"param": "value"}'

    def test_handle_function_call_arguments_done_event(self):
        """测试 FunctionCallArgumentsDoneEvent 处理"""
        processor = ResponseEventProcessor()

        # 先添加一个函数调用消息
        processor._messages.append(
            {
                "type": "function_call",
                "arguments": "partial",
            },
        )

        # 使用 Mock 来模拟 FunctionCallArgumentsDoneEvent
        event = Mock(spec=FunctionCallArgumentsDoneEvent)
        event.arguments = '{"param": "value"}'
        # Mock 必须有 type 属性才能通过初始检查
        event.type = "other"

        result = processor.handle_event(event)

        assert result == []
        assert processor._messages[0]["arguments"] == '{"param": "value"}'

    def test_handle_response_completed_event(self):
        """测试 ResponseCompletedEvent 处理"""
        processor = ResponseEventProcessor()
        # 设置时间以便测试时间计算
        processor._start_time = datetime.now(timezone.utc)
        processor._first_output_time = datetime.now(timezone.utc)
        processor._output_complete_time = datetime.now(timezone.utc)

        # 使用 Mock 来模拟 ResponseCompletedEvent
        mock_usage = Mock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50

        mock_response = Mock()
        mock_response.usage = mock_usage

        event = Mock(spec=ResponseCompletedEvent)
        event.response = mock_response
        # Mock 必须有 type 属性才能通过初始检查
        event.type = "other"

        result = processor.handle_event(event)

        assert len(result) == 2  # UsageEvent 和 TimingEvent
        assert isinstance(result[0], UsageEvent)
        assert isinstance(result[1], TimingEvent)

        # 检查使用量数据
        assert result[0].usage.input_tokens == 100
        assert result[0].usage.output_tokens == 50

        # 检查时间数据
        assert result[1].timing.latency_ms is not None
        assert result[1].timing.output_time_ms is not None

        # 检查内部状态
        assert processor._usage_time is not None
        assert processor._usage_data["input_tokens"] == 100
        assert processor._usage_data["output_tokens"] == 50

    def test_handle_response_completed_event_no_usage(self):
        """测试 ResponseCompletedEvent 无使用量数据的处理"""
        processor = ResponseEventProcessor()

        # 使用 Mock 来模拟没有使用量的 ResponseCompletedEvent
        mock_response = Mock()
        mock_response.usage = None

        event = Mock(spec=ResponseCompletedEvent)
        event.response = mock_response
        # Mock 必须有 type 属性才能通过初始检查
        event.type = "other"

        result = processor.handle_event(event)

        assert result == []

    def test_messages_property(self):
        """测试 messages 属性"""
        processor = ResponseEventProcessor()

        # 初始状态
        assert processor.messages == []

        # 添加消息
        test_message = {"test": "message"}
        processor._messages.append(test_message)

        assert len(processor.messages) == 1
        assert processor.messages[0] == test_message

    def test_reset_method(self):
        """测试 reset 方法"""
        processor = ResponseEventProcessor()

        # 设置一些状态
        processor._messages = [{"test": "message"}]
        processor._start_time = datetime.now(timezone.utc)
        processor._first_output_time = datetime.now(timezone.utc)
        processor._output_complete_time = datetime.now(timezone.utc)
        processor._usage_time = datetime.now(timezone.utc)
        processor._usage_data = {"input_tokens": 100}

        # 重置
        processor.reset()

        # 验证所有状态都被重置
        assert processor._messages == []
        assert processor._start_time is None
        assert processor._first_output_time is None
        assert processor._output_complete_time is None
        assert processor._usage_time is None
        assert processor._usage_data == {}

    def test_edge_case_empty_messages(self):
        """测试空消息列表的边界情况"""
        processor = ResponseEventProcessor()

        # 测试在没有消息时处理内容相关事件
        event = ContentPartAddedEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
            item_id="item_123",
            output_index=0,
            content_index=0,
            part={"type": "text", "text": "test"},
        )
        result = processor.handle_event(event)
        assert result == []

        # 测试在没有消息时处理文本增量事件
        event = OutputTextDeltaEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            item_id="item_123",
            output_index=0,
            content_index=0,
            delta="test",
        )
        result = processor.handle_event(event)
        assert result == []

    def test_edge_case_non_list_content(self):
        """测试非列表内容的边界情况"""
        processor = ResponseEventProcessor()

        # 添加一个内容不是列表的消息
        processor._messages.append({"content": "not a list"})

        # 测试处理 ContentPartAddedEvent
        event = ContentPartAddedEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
            item_id="item_123",
            output_index=0,
            content_index=0,
            part={"type": "text", "text": "test"},
        )
        result = processor.handle_event(event)
        assert result == []

        # 测试处理 OutputTextDeltaEvent
        event = OutputTextDeltaEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            item_id="item_123",
            output_index=0,
            content_index=0,
            delta="test",
        )
        result = processor.handle_event(event)
        assert result == []

    def test_timing_calculations(self):
        """测试时间计算逻辑"""
        processor = ResponseEventProcessor()

        # 手动设置时间以测试计算
        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        first_output_time = datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)  # 1秒后
        output_complete_time = datetime(2023, 1, 1, 12, 0, 3, tzinfo=timezone.utc)  # 3秒后

        processor._start_time = start_time
        processor._first_output_time = first_output_time
        processor._output_complete_time = output_complete_time

        # 使用 Mock 来模拟 ResponseCompletedEvent
        mock_usage = Mock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50

        mock_response = Mock()
        mock_response.usage = mock_usage

        event = Mock(spec=ResponseCompletedEvent)
        event.response = mock_response
        # Mock 必须有 type 属性才能通过初始检查
        event.type = "other"

        result = processor.handle_event(event)

        # 验证时间计算
        timing_event = result[1]  # 第二个是 TimingEvent
        assert isinstance(timing_event, TimingEvent)
        assert timing_event.timing.latency_ms == 1000  # 1秒 = 1000毫秒
        assert timing_event.timing.output_time_ms == 2000  # 2秒 = 2000毫秒

    def test_unknown_event_type(self):
        """测试未知事件类型"""
        processor = ResponseEventProcessor()

        # 创建一个未知类型的事件
        unknown_event = Mock(spec=[])
        unknown_event.type = "unknown_type"

        result = processor.handle_event(unknown_event)
        assert result == []
