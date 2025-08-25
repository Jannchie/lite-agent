"""Unit tests for Response API format - updated for new NewMessage-only architecture."""


from lite_agent import Agent, Runner
from lite_agent.types import NewUserMessage, ResponseInputImage, ResponseInputText


class TestResponseAPIFormatNew:
    """Test Response API format handling after removing dict support."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = Agent(
            model="gpt-4.1-nano",
            name="TestAgent",
            instructions="You are a helpful assistant.",
        )
        self.runner = Runner(self.agent)

    def test_response_input_text_creation(self):
        """测试 ResponseInputText 对象创建"""
        text_input = ResponseInputText(
            type="input_text",
            text="Hello world",
        )

        assert text_input.type == "input_text"
        assert text_input.text == "Hello world"

    def test_response_input_image_creation_with_url(self):
        """测试使用 URL 创建 ResponseInputImage"""
        image_input = ResponseInputImage(
            type="input_image",
            detail="high",
            image_url="https://example.com/test.jpg",
        )

        assert image_input.type == "input_image"
        assert image_input.detail == "high"
        assert image_input.image_url == "https://example.com/test.jpg"
        assert image_input.file_id is None

    def test_response_input_image_creation_with_file_id(self):
        """测试使用 file_id 创建 ResponseInputImage"""
        image_input = ResponseInputImage(
            type="input_image",
            detail="auto",
            file_id="file-12345",
        )

        assert image_input.type == "input_image"
        assert image_input.detail == "auto"
        assert image_input.file_id == "file-12345"
        assert image_input.image_url is None

    def test_dict_format_is_converted(self):
        """测试dict格式被正确转换"""
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

        # Dict format should now be converted to NewMessage
        self.runner.append_message(
            {
                "role": "user",
                "content": mixed_content,
            },
        )

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)

    def test_dict_content_is_converted(self):
        """测试使用字典内容格式被正确转换"""
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

        # Dict format should now be converted
        self.runner.append_message(
            {
                "role": "user",
                "content": dict_content,
            },
        )

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)

    def test_new_message_format_works(self):
        """测试使用NewMessage格式正常工作"""
        from lite_agent.types import NewUserMessage, UserTextContent

        # Create message using the new format
        message = NewUserMessage(content=[UserTextContent(text="Hello!")])
        self.runner.append_message(message)

        assert len(self.runner.messages) == 1
        assert isinstance(self.runner.messages[0], NewUserMessage)
        from lite_agent.types import UserTextContent
        assert isinstance(self.runner.messages[0].content[0], UserTextContent)
        assert self.runner.messages[0].content[0].text == "Hello!"

    def test_convenience_methods_work(self):
        """测试便捷方法正常工作"""
        self.runner.add_user_message("Hello!")
        self.runner.add_assistant_message("Hi there!")
        self.runner.add_system_message("Be helpful.")

        messages = self.runner.get_messages()
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "system"
