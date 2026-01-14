"""Tests for PromptBuilder module."""

import pytest

from src.prompt import PromptBuilder, PromptConfig, PromptContext


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()
        assert config.max_examples == 3
        assert config.include_schema is True
        assert config.include_examples is True
        assert config.max_example_length == 500

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = PromptConfig(
            max_examples=5,
            include_schema=False,
            max_example_length=1000,
        )
        assert config.max_examples == 5
        assert config.include_schema is False
        assert config.max_example_length == 1000


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    @pytest.mark.unit
    def test_build_without_store(self):
        """Test building prompt without VectorStore."""
        builder = PromptBuilder()
        prompt = builder.build("login form")

        assert "UI Component Schema" in prompt
        assert "login form" in prompt
        assert "Generate a UI layout" in prompt

    @pytest.mark.unit
    def test_build_schema_only(self):
        """Test building prompt with schema only."""
        config = PromptConfig(include_examples=False)
        builder = PromptBuilder(config=config)
        prompt = builder.build("dashboard")

        assert "UI Component Schema" in prompt
        assert "Similar Layout Examples" not in prompt

    @pytest.mark.unit
    def test_build_no_schema(self):
        """Test building prompt without schema."""
        config = PromptConfig(include_schema=False, include_examples=False)
        builder = PromptBuilder(config=config)
        prompt = builder.build("settings page")

        assert "UI Component Schema" not in prompt
        assert "Your Task" in prompt

    @pytest.mark.unit
    def test_build_with_context(self):
        """Test build_with_context returns context info."""
        builder = PromptBuilder()
        prompt, context = builder.build_with_context("navigation menu")

        assert context.query == "navigation menu"
        assert context.schema_included is True
        assert context.example_count == 0  # No store
        assert context.total_tokens_estimate > 0


class TestPromptContext:
    """Tests for PromptContext dataclass."""

    @pytest.mark.unit
    def test_context_creation(self):
        """Test creating a prompt context."""
        context = PromptContext(
            query="test query",
            schema_included=True,
            example_count=3,
            example_ids=["a", "b", "c"],
        )
        assert context.query == "test query"
        assert context.schema_included is True
        assert context.example_count == 3
        assert len(context.example_ids) == 3
