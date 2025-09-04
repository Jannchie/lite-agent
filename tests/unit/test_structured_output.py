"""Unit tests for structured output functionality."""

from pydantic import BaseModel, Field

from lite_agent import Agent
from lite_agent.client import _prepare_response_format


class PersonSchema(BaseModel):
    """Test response format model."""

    name: str = Field(description="Name field")
    value: int = Field(description="Value field", ge=0)


class TestPrepareResponseFormat:
    """Test the _prepare_response_format function."""

    def test_prepare_response_format_none(self):
        """Test with None input."""
        result = _prepare_response_format(None)
        assert result is None

    def test_prepare_response_format_pydantic_model(self):
        """Test with Pydantic model class."""
        result = _prepare_response_format(PersonSchema)

        assert result is not None
        assert isinstance(result, dict)
        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["name"] == "PersonSchema"
        assert result["json_schema"]["strict"] is True
        assert "schema" in result["json_schema"]

        # Check schema contains expected fields
        schema = result["json_schema"]["schema"]
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "value" in schema["properties"]

    def test_prepare_response_format_dict(self):
        """Test with dictionary format."""
        format_dict = {"type": "json_schema", "json_schema": {"name": "CustomFormat", "schema": {"type": "object", "properties": {}}, "strict": True}}

        result = _prepare_response_format(format_dict)
        assert result == format_dict

    def test_prepare_response_format_invalid_type(self):
        """Test with invalid type."""
        result = _prepare_response_format("invalid")  # type: ignore[arg-type]
        assert result is None


class TestAgentStructuredOutput:
    """Test Agent class structured output functionality."""

    def test_agent_init_with_response_format(self):
        """Test Agent initialization with response_format."""
        agent = Agent(
            model="gpt-4o-mini",
            name="TestAgent",
            instructions="Test instructions",
            response_format=PersonSchema,
        )

        assert agent.response_format == PersonSchema

    def test_agent_init_without_response_format(self):
        """Test Agent initialization without response_format."""
        agent = Agent(
            model="gpt-4o-mini",
            name="TestAgent",
            instructions="Test instructions",
        )

        assert agent.response_format is None

    def test_agent_set_response_format(self):
        """Test setting response format after initialization."""
        agent = Agent(
            model="gpt-4o-mini",
            name="TestAgent",
            instructions="Test instructions",
        )

        agent.set_response_format(PersonSchema)
        assert agent.response_format == PersonSchema

        agent.set_response_format(None)
        assert agent.response_format is None

    def test_agent_get_response_format(self):
        """Test getting current response format."""
        agent = Agent(
            model="gpt-4o-mini",
            name="TestAgent",
            instructions="Test instructions",
            response_format=PersonSchema,
        )

        assert agent.get_response_format() == PersonSchema


class PersonSchemaValidation:
    """Test response format validation."""

    def test_valid_pydantic_model(self):
        """Test with valid Pydantic model."""
        # This should not raise any errors
        result = _prepare_response_format(PersonSchema)
        assert result is not None

    def test_non_pydantic_class(self):
        """Test with non-Pydantic class."""

        class RegularClass:
            pass

        # Should return None for non-Pydantic classes
        result = _prepare_response_format(RegularClass)  # type: ignore[arg-type]
        assert result is None

    def test_valid_dict_format(self):
        """Test with valid dictionary format."""
        valid_dict = {"type": "json_schema", "json_schema": {"name": "TestFormat", "schema": {"type": "object", "properties": {"test": {"type": "string"}}}, "strict": True}}

        result = _prepare_response_format(valid_dict)
        assert result == valid_dict
