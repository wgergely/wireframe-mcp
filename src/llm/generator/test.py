"""Tests for LLM generator module.

Covers:
- RetryStrategy: JSON repair and retry logic
- LayoutGenerator: Orchestration with mocked backend
"""

import json
from unittest.mock import MagicMock

import pytest

from src.ir import TranspilationContext

from ..backend.base import GenerationResult, InvalidResponseError
from .lib import GenerationOutput, GenerationStats, GeneratorConfig, LayoutGenerator
from .retry import RetryConfig, RetryStrategy

# =============================================================================
# RetryStrategy Tests
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.repair_json is True
        assert config.exponential_backoff is True
        assert config.initial_delay == 1.0
        assert config.max_delay == 30.0

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_retries=5,
            repair_json=False,
            initial_delay=2.0,
        )
        assert config.max_retries == 5
        assert config.repair_json is False
        assert config.initial_delay == 2.0


class TestRetryStrategyJsonRepair:
    """Tests for JSON repair functionality."""

    @pytest.fixture
    def strategy(self):
        """Create a RetryStrategy with repair enabled."""
        return RetryStrategy(RetryConfig(repair_json=True))

    @pytest.mark.unit
    def test_repair_markdown_code_blocks(self, strategy):
        """Test removing markdown code blocks."""
        content = '```json\n{"id": "test", "type": "container"}\n```'
        result = strategy.repair_json(content)
        assert result is not None
        assert result["id"] == "test"
        assert result["type"] == "container"

    @pytest.mark.unit
    def test_repair_trailing_comma_object(self, strategy):
        """Test fixing trailing comma in object."""
        content = '{"id": "test", "type": "container",}'
        result = strategy.repair_json(content)
        assert result is not None
        assert result["id"] == "test"

    @pytest.mark.unit
    def test_repair_trailing_comma_array(self, strategy):
        """Test fixing trailing comma in array."""
        content = '{"children": ["a", "b", "c",]}'
        result = strategy.repair_json(content)
        assert result is not None
        assert result["children"] == ["a", "b", "c"]

    @pytest.mark.unit
    def test_repair_extract_from_mixed(self, strategy):
        """Test extracting JSON from mixed content."""
        content = 'Here is your JSON:\n{"id": "root"}\nHope this helps!'
        result = strategy.repair_json(content)
        assert result is not None
        assert result["id"] == "root"

    @pytest.mark.unit
    def test_repair_with_prefix(self, strategy):
        """Test removing common prefixes."""
        content = 'Here is the JSON:\n{"id": "test"}'
        result = strategy.repair_json(content)
        assert result is not None
        assert result["id"] == "test"

    @pytest.mark.unit
    def test_repair_balanced_braces(self, strategy):
        """Test extracting with balanced braces."""
        content = 'Some text before {"id": "test"} some text after'
        result = strategy.repair_json(content)
        assert result is not None
        assert result["id"] == "test"

    @pytest.mark.unit
    def test_repair_disabled(self):
        """Test that repair does nothing when disabled."""
        strategy = RetryStrategy(RetryConfig(repair_json=False))
        content = '```json\n{"id": "test"}\n```'
        result = strategy.repair_json(content)
        assert result is None

    @pytest.mark.unit
    def test_repair_valid_json(self, strategy):
        """Test that valid JSON passes through."""
        content = '{"id": "test", "type": "container"}'
        result = strategy.repair_json(content)
        assert result is not None
        assert result["id"] == "test"


class TestRetryStrategyBackoff:
    """Tests for backoff delay calculation."""

    @pytest.mark.unit
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        strategy = RetryStrategy(RetryConfig(initial_delay=1.0, max_delay=30.0))
        assert strategy.get_backoff_delay(0) == 1.0
        assert strategy.get_backoff_delay(1) == 2.0
        assert strategy.get_backoff_delay(2) == 4.0
        assert strategy.get_backoff_delay(3) == 8.0

    @pytest.mark.unit
    def test_backoff_max_delay(self):
        """Test that delay is capped at max_delay."""
        strategy = RetryStrategy(RetryConfig(initial_delay=1.0, max_delay=10.0))
        assert strategy.get_backoff_delay(10) == 10.0  # 2^10 = 1024, capped at 10

    @pytest.mark.unit
    def test_linear_backoff(self):
        """Test no exponential backoff when disabled."""
        strategy = RetryStrategy(
            RetryConfig(exponential_backoff=False, initial_delay=2.0)
        )
        assert strategy.get_backoff_delay(0) == 2.0
        assert strategy.get_backoff_delay(5) == 2.0


class TestRetryStrategyDecision:
    """Tests for retry decision logic."""

    @pytest.fixture
    def strategy(self):
        """Create a RetryStrategy with default config."""
        return RetryStrategy(RetryConfig(max_retries=3))

    @pytest.mark.unit
    def test_should_retry_json_error(self, strategy):
        """Test that JSON decode errors are retryable."""
        error = json.JSONDecodeError("Expecting value", "", 0)
        assert strategy.should_retry(error, 0) is True

    @pytest.mark.unit
    def test_should_retry_rate_limit(self, strategy):
        """Test that rate limit errors are retryable."""
        error = Exception("rate limit exceeded")
        assert strategy.should_retry(error, 0) is True

    @pytest.mark.unit
    def test_should_retry_timeout(self, strategy):
        """Test that timeout errors are retryable."""
        error = Exception("connection timeout")
        assert strategy.should_retry(error, 0) is True

    @pytest.mark.unit
    def test_should_not_retry_max_attempts(self, strategy):
        """Test that retries stop at max attempts."""
        error = json.JSONDecodeError("test", "", 0)
        assert strategy.should_retry(error, 3) is False

    @pytest.mark.unit
    def test_should_not_retry_unknown_error(self, strategy):
        """Test that unknown errors are not retried."""
        error = Exception("unknown authentication failure")
        assert strategy.should_retry(error, 0) is False


# =============================================================================
# LayoutGenerator Tests (Mocked Backend)
# =============================================================================


class TestGeneratorConfig:
    """Tests for GeneratorConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = GeneratorConfig()
        assert config.max_retries == 3
        assert config.retry_on_validation_error is True
        assert config.repair_json is True
        assert config.temperature == 0.7
        assert config.include_rag_context is True
        assert config.max_rag_examples == 3

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = GeneratorConfig(
            max_retries=5,
            temperature=0.3,
            include_rag_context=False,
        )
        assert config.max_retries == 5
        assert config.temperature == 0.3
        assert config.include_rag_context is False


class TestGenerationStats:
    """Tests for GenerationStats dataclass."""

    @pytest.mark.unit
    def test_default_stats(self):
        """Test default statistics values."""
        stats = GenerationStats()
        assert stats.attempts == 0
        assert stats.validation_retries == 0
        assert stats.json_repairs == 0
        assert stats.total_tokens == 0
        assert stats.final_model == ""


class TestLayoutGeneratorMocked:
    """Tests for LayoutGenerator with mocked backend."""

    @pytest.fixture
    def valid_layout_json(self):
        """Valid LayoutNode JSON response."""
        return json.dumps(
            {
                "id": "root",
                "type": "container",
                "label": "Login Form",
                "orientation": "vertical",
                "children": [
                    {"id": "email", "type": "input", "label": "Email"},
                    {"id": "password", "type": "input", "label": "Password"},
                    {"id": "submit", "type": "button", "label": "Login"},
                ],
            }
        )

    @pytest.fixture
    def mock_backend(self, valid_layout_json):
        """Create a mock LLM backend."""
        backend = MagicMock()
        backend.provider = "mock"
        backend.model_name = "mock-model"
        backend.supports_json_mode = True
        backend.generate.return_value = GenerationResult(
            content=valid_layout_json,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock-model",
        )
        return backend

    @pytest.mark.unit
    def test_generate_success(self, mock_backend, valid_layout_json):
        """Test successful layout generation."""
        generator = LayoutGenerator(backend=mock_backend)
        output = generator.generate("login form with email and password")

        assert isinstance(output, GenerationOutput)
        assert isinstance(output.context, TranspilationContext)
        assert output.context.node.id == "root"
        assert len(output.context.node.children) == 3
        assert output.stats.attempts == 1
        assert output.stats.total_tokens == 150

    @pytest.mark.unit
    def test_generate_returns_context(self, mock_backend):
        """Test that generation returns proper TranspilationContext."""
        generator = LayoutGenerator(backend=mock_backend)
        output = generator.generate("dashboard", target_provider="plantuml")

        assert output.context.target_provider == "plantuml"
        assert output.context.node is not None
        assert output.prompt_context.query == "dashboard"

    @pytest.mark.unit
    def test_generate_with_json_repair(self, mock_backend):
        """Test generation with JSON repair."""
        # Return malformed JSON
        mock_backend.generate.return_value = GenerationResult(
            content='```json\n{"id": "root", "type": "container"}\n```',
            finish_reason="stop",
            usage={"total_tokens": 100},
            model="mock-model",
        )

        generator = LayoutGenerator(
            backend=mock_backend,
            config=GeneratorConfig(repair_json=True),
        )
        output = generator.generate("simple container")

        assert output.context.node.id == "root"
        assert output.stats.json_repairs == 1

    @pytest.mark.unit
    def test_generate_max_retries_exhausted(self, mock_backend):
        """Test that generation fails after max retries."""
        mock_backend.generate.return_value = GenerationResult(
            content="not valid json at all",
            finish_reason="stop",
            usage={"total_tokens": 100},
            model="mock-model",
        )

        generator = LayoutGenerator(
            backend=mock_backend,
            config=GeneratorConfig(max_retries=2, repair_json=True),
        )

        with pytest.raises(InvalidResponseError, match="Failed to generate"):
            generator.generate("anything")

    @pytest.mark.unit
    def test_generate_validation_retry(self, mock_backend):
        """Test retry with validation error feedback."""
        # First call returns invalid (duplicate IDs), second returns valid
        invalid_layout = json.dumps(
            {
                "id": "root",
                "type": "container",
                "children": [
                    {"id": "same", "type": "button"},
                    {"id": "same", "type": "button"},  # Duplicate ID
                ],
            }
        )
        valid_layout = json.dumps(
            {
                "id": "root",
                "type": "container",
                "children": [
                    {"id": "btn1", "type": "button"},
                    {"id": "btn2", "type": "button"},
                ],
            }
        )

        mock_backend.generate.side_effect = [
            GenerationResult(
                content=invalid_layout,
                finish_reason="stop",
                usage={"total_tokens": 100},
                model="mock-model",
            ),
            GenerationResult(
                content=valid_layout,
                finish_reason="stop",
                usage={"total_tokens": 100},
                model="mock-model",
            ),
        ]

        generator = LayoutGenerator(
            backend=mock_backend,
            config=GeneratorConfig(retry_on_validation_error=True),
        )
        output = generator.generate("two buttons")

        assert output.stats.attempts == 2
        assert output.stats.validation_retries == 1
        assert output.context.node.children[0].id == "btn1"


# =============================================================================
# Integration Tests (requires real API key)
# =============================================================================


@pytest.mark.integration
class TestLayoutGeneratorIntegration:
    """Integration tests with real OpenAI API."""

    def test_real_openai_generation(self):
        """Test actual generation with OpenAI API."""
        generator = LayoutGenerator()
        output = generator.generate("simple login form")

        assert output.context.node is not None
        assert output.context.node.id
        assert output.stats.total_tokens > 0
        print(f"Generated layout: {output.context.node}")
        print(f"Stats: {output.stats}")
