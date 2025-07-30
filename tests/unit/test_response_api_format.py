"""
测试新的 Response API 格式支持
"""

import pytest

from lite_agent.agent import Agent
from lite_agent.runner import Runner
from lite_agent.types import ResponseInputImage, ResponseInputText


class TestResponseAPIFormat:
    """测试新的 Response API 格式支持"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.agent = Agent(
            model="gpt-4-turbo",
            name="TestAgent",
            instructions="Test agent for Response API format",
        )
        self.runner = Runner(self.agent)

    def test_response_input_text_creation(self):
        """测试 ResponseInputText 的创建"""
        text_input = ResponseInputText(
            type="input_text",
            text="Hello, world!",
        )

        assert text_input.type == "input_text"
        assert text_input.text == "Hello, world!"

    def test_response_input_image_creation_with_url(self):
        """测试使用 URL 的 ResponseInputImage 创建"""
        image_input = ResponseInputImage(
            type="input_image",
            detail="high",
            image_url="https://example.com/image.jpg",
        )

        assert image_input.type == "input_image"
        assert image_input.detail == "high"
        assert image_input.image_url == "https://example.com/image.jpg"
        assert image_input.file_id is None

    def test_response_input_image_creation_with_file_id(self):
        """测试使用 file_id 的 ResponseInputImage 创建"""
        image_input = ResponseInputImage(
            type="input_image",
            detail="auto",
            file_id="file-12345",
        )

        assert image_input.type == "input_image"
        assert image_input.detail == "auto"
        assert image_input.file_id == "file-12345"
        assert image_input.image_url is None

    def test_append_message_with_response_api_format(self):
        """测试使用新的 Response API 格式添加消息"""
        mixed_content = [
            ResponseInputText(
                type="input_text",
                text="What's in this image?",
            ),
            ResponseInputImage(
                type="input_image",
                detail="high",
                image_url="https://example.com/test.jpg",
            ),
        ]

        self.runner.append_message(
            {
                "role": "user",
                "content": mixed_content,
            },
        )

        assert len(self.runner.messages) == 1
        message = self.runner.messages[0]
        assert hasattr(message, "role")
        assert message.role == "user"  # type: ignore
        assert hasattr(message, "content")
        assert isinstance(message.content, list)  # type: ignore
        assert len(message.content) == 2  # type: ignore

        # 检查第一个内容项 (text) - 现在使用新格式的类型名称
        if hasattr(message, "content"):
            text_item = message.content[0]  # type: ignore
            assert text_item.type == "text"
            assert text_item.text == "What's in this image?"

            # 检查第二个内容项 (image) - 现在使用新格式的类型名称
            image_item = message.content[1]  # type: ignore
            assert image_item.type == "image"
            assert image_item.detail == "high"
            assert image_item.image_url == "https://example.com/test.jpg"

    def test_append_message_with_dict_format(self):
        """测试使用字典格式的 Response API 内容"""
        dict_content = [
            {
                "type": "input_text",
                "text": "What's in this image?",
            },
            {
                "type": "input_image",
                "detail": "high",
                "image_url": "https://example.com/test.jpg",
            },
        ]

        self.runner.append_message(
            {
                "role": "user",
                "content": dict_content,
            },
        )

        assert len(self.runner.messages) == 1
        message = self.runner.messages[0]
        assert hasattr(message, "role")
        assert message.role == "user"  # type: ignore
        assert hasattr(message, "content")
        assert isinstance(message.content, list)  # type: ignore
        assert len(message.content) == 2  # type: ignore

        # 检查内容被正确转换为 Pydantic 对象 - 现在使用新格式的类型名称
        if hasattr(message, "content"):
            text_item = message.content[0]  # type: ignore
            assert text_item.type == "text"
            assert text_item.text == "What's in this image?"

            image_item = message.content[1]  # type: ignore
            assert image_item.type == "image"
            assert image_item.detail == "high"
            assert image_item.image_url == "https://example.com/test.jpg"

    def test_conversion_to_completion_api_format(self):
        """测试转换为 Completion API 格式"""
        mixed_content = [
            ResponseInputText(
                type="input_text",
                text="Analyze this image",
            ),
            ResponseInputImage(
                type="input_image",
                detail="low",
                image_url="https://example.com/photo.png",
            ),
        ]

        self.runner.append_message(
            {
                "role": "user",
                "content": mixed_content,
            },
        )

        # 转换为 Completion API 格式 - 需要使用 legacy_messages
        converted_messages = self.agent._convert_responses_to_completions_format(self.runner.legacy_messages)

        assert len(converted_messages) == 1
        converted_msg = converted_messages[0]
        assert converted_msg["role"] == "user"
        assert isinstance(converted_msg["content"], list)
        assert len(converted_msg["content"]) == 2

        # 检查转换后的文本内容
        text_content = converted_msg["content"][0]
        assert text_content["type"] == "text"
        assert text_content["text"] == "Analyze this image"

        # 检查转换后的图像内容
        image_content = converted_msg["content"][1]
        assert image_content["type"] == "image_url"
        assert image_content["image_url"]["url"] == "https://example.com/photo.png"
        assert image_content["image_url"]["detail"] == "low"

    def test_dict_format_conversion_to_completion_api(self):
        """测试字典格式转换为 Completion API 格式"""
        dict_content = [
            {
                "type": "input_text",
                "text": "Describe this photo",
            },
            {
                "type": "input_image",
                "image_url": "https://example.com/photo.jpg",
            },
        ]

        self.runner.append_message(
            {
                "role": "user",
                "content": dict_content,
            },
        )

        converted_messages = self.agent._convert_responses_to_completions_format(self.runner.legacy_messages)

        assert len(converted_messages) == 1
        converted_msg = converted_messages[0]
        assert converted_msg["role"] == "user"
        assert isinstance(converted_msg["content"], list)
        assert len(converted_msg["content"]) == 2

        # 检查文本转换
        text_content = converted_msg["content"][0]
        assert text_content["type"] == "text"
        assert text_content["text"] == "Describe this photo"

        # 检查图像转换
        image_content = converted_msg["content"][1]
        assert image_content["type"] == "image_url"
        assert image_content["image_url"]["url"] == "https://example.com/photo.jpg"
        # detail 应该是默认的 "auto"
        assert image_content["image_url"]["detail"] == "auto"

    def test_conversion_with_file_id_raises_error(self):
        """测试 file_id 会抛出异常"""
        file_content = [
            ResponseInputImage(
                type="input_image",
                detail="auto",
                file_id="file-abc123",
            ),
        ]

        self.runner.append_message(
            {
                "role": "user",
                "content": file_content,
            },
        )

        # 应该抛出 ValueError 异常
        with pytest.raises(ValueError, match="File ID input is not supported for Completion API"):
            self.agent._convert_responses_to_completions_format(self.runner.legacy_messages)

    def test_conversion_with_missing_image_data_raises_error(self):
        """测试既没有 image_url 也没有 file_id时会抛出异常"""
        # 应该在创建对象时就抛出异常
        with pytest.raises(ValueError, match="ResponseInputImage must have either file_id or image_url"):
            ResponseInputImage(
                type="input_image",
                detail="auto",
                # 既没有 file_id 也没有 image_url
            )

    def test_mixed_with_legacy_format(self):
        """测试新格式与传统格式的混合使用"""
        # 添加新格式消息
        new_format_content = [
            ResponseInputText(
                type="input_text",
                text="New format message",
            ),
        ]

        self.runner.append_message(
            {
                "role": "user",
                "content": new_format_content,
            },
        )

        # 添加传统格式消息
        self.runner.append_message(
            {
                "role": "user",
                "content": "Traditional text message",
            },
        )

        assert len(self.runner.messages) == 2

        # 检查两种格式都被正确处理
        converted_messages = self.agent._convert_responses_to_completions_format(self.runner.legacy_messages)
        assert len(converted_messages) == 2

        # 新格式消息的转换 - 单一文本内容被优化为字符串
        assert isinstance(converted_messages[0]["content"], str)
        assert converted_messages[0]["content"] == "New format message"

        # 传统格式消息保持不变
        assert isinstance(converted_messages[1]["content"], str)
        assert converted_messages[1]["content"] == "Traditional text message"
