"""Tests for LLM backend implementations."""

import os

import pytest

from .base import (
    AuthenticationError,
    GenerationConfig,
    GenerationResult,
)
from .factory import create_llm_backend
from .model_spec import (
    LLMCapability,
    LLMModel,
    LLMProviderType,
    LLMSpec,
    get_llm_spec,
)


class TestLLMSpec:
    """Tests for LLMSpec dataclass."""

    @pytest.mark.unit
    def test_spec_creation(self):
        """Test creating an LLMSpec."""
        spec = LLMSpec(
            name="test-model",
            provider=LLMProviderType.OPENAI,
            context_window=128000,
            max_output_tokens=4096,
        )
        assert spec.name == "test-model"
        assert spec.provider == LLMProviderType.OPENAI
        assert spec.context_window == 128000
        assert spec.max_output_tokens == 4096

    @pytest.mark.unit
    def test_spec_capabilities(self):
        """Test capability checking."""
        spec = LLMSpec(
            name="test",
            provider=LLMProviderType.OPENAI,
            context_window=128000,
            max_output_tokens=4096,
            capabilities=frozenset(
                {
                    LLMCapability.JSON_MODE,
                    LLMCapability.STREAMING,
                }
            ),
        )
        assert spec.supports(LLMCapability.JSON_MODE)
        assert spec.supports(LLMCapability.STREAMING)
        assert not spec.supports(LLMCapability.VISION)

    @pytest.mark.unit
    def test_spec_is_local(self):
        """Test is_local property."""
        ollama_spec = LLMSpec(
            name="test",
            provider=LLMProviderType.OLLAMA,
            context_window=32000,
            max_output_tokens=4096,
        )
        openai_spec = LLMSpec(
            name="test",
            provider=LLMProviderType.OPENAI,
            context_window=128000,
            max_output_tokens=4096,
        )
        assert ollama_spec.is_local
        assert not ollama_spec.is_remote
        assert not openai_spec.is_local
        assert openai_spec.is_remote


class TestLLMModel:
    """Tests for LLMModel enum registry."""

    @pytest.mark.unit
    def test_openai_models_exist(self):
        """Test that OpenAI models are defined."""
        assert LLMModel.GPT_4_1_MINI.spec.name == "gpt-4.1-mini"
        assert LLMModel.GPT_4_1_MINI.spec.requires_api_key
        assert LLMModel.GPT_4_1_MINI.spec.provider == LLMProviderType.OPENAI

    @pytest.mark.unit
    def test_anthropic_models_exist(self):
        """Test that Anthropic models are defined."""
        assert LLMModel.CLAUDE_SONNET_4_5.spec.provider == LLMProviderType.ANTHROPIC
        assert LLMModel.CLAUDE_SONNET_4_5.spec.name == "claude-sonnet-4-5"
        assert LLMModel.CLAUDE_SONNET_4_5.spec.context_window == 200000

    @pytest.mark.unit
    def test_deepseek_models_exist(self):
        """Test that DeepSeek models are defined."""
        assert LLMModel.DEEPSEEK_V3_2.spec.provider == LLMProviderType.DEEPSEEK
        assert "deepseek.com" in (LLMModel.DEEPSEEK_V3_2.spec.base_url or "")

    @pytest.mark.unit
    def test_qwen_models_exist(self):
        """Test that Qwen models are defined."""
        assert LLMModel.QWEN_TURBO.spec.provider == LLMProviderType.QWEN
        assert "dashscope" in (LLMModel.QWEN_TURBO.spec.base_url or "")

    @pytest.mark.unit
    def test_ollama_models_exist(self):
        """Test that Ollama models are defined."""
        assert LLMModel.OLLAMA_QWEN3.spec.provider == LLMProviderType.OLLAMA
        assert not LLMModel.OLLAMA_QWEN3.spec.requires_api_key
        assert LLMModel.OLLAMA_QWEN3.spec.is_local

    @pytest.mark.unit
    def test_by_name_lookup(self):
        """Test looking up models by name."""
        model = LLMModel.by_name("gpt-4.1-mini")
        assert model == LLMModel.GPT_4_1_MINI

        model = LLMModel.by_name("claude-sonnet-4-5")
        assert model == LLMModel.CLAUDE_SONNET_4_5

        model = LLMModel.by_name("nonexistent")
        assert model is None

    @pytest.mark.unit
    def test_list_by_provider(self):
        """Test listing models by provider."""
        openai_models = LLMModel.list_by_provider(LLMProviderType.OPENAI)
        assert len(openai_models) >= 3
        assert all(m.spec.provider == LLMProviderType.OPENAI for m in openai_models)

        ollama_models = LLMModel.list_by_provider(LLMProviderType.OLLAMA)
        assert len(ollama_models) >= 3
        assert all(m.spec.is_local for m in ollama_models)

    @pytest.mark.unit
    def test_list_local_remote(self):
        """Test listing local vs remote models."""
        local_models = LLMModel.list_local()
        remote_models = LLMModel.list_remote()

        assert all(m.spec.is_local for m in local_models)
        assert all(m.spec.is_remote for m in remote_models)
        assert len(local_models) + len(remote_models) == len(LLMModel)


class TestGetLLMSpec:
    """Tests for get_llm_spec helper."""

    @pytest.mark.unit
    def test_from_string(self):
        """Test resolving from string name."""
        spec = get_llm_spec("gpt-4.1-mini")
        assert spec.name == "gpt-4.1-mini"

    @pytest.mark.unit
    def test_from_enum(self):
        """Test resolving from enum."""
        spec = get_llm_spec(LLMModel.GPT_4_1_MINI)
        assert spec.name == "gpt-4.1-mini"

    @pytest.mark.unit
    def test_from_spec(self):
        """Test resolving from LLMSpec."""
        original = LLMModel.GPT_4_1_MINI.spec
        spec = get_llm_spec(original)
        assert spec is original

    @pytest.mark.unit
    def test_unknown_model_raises(self):
        """Test that unknown model names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_llm_spec("nonexistent-model")


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    @pytest.mark.unit
    def test_defaults(self):
        """Test default configuration."""
        config = GenerationConfig()
        assert config.temperature == 0.7
        assert config.json_mode is True
        assert config.max_tokens == 4096
        assert config.top_p == 1.0
        assert config.seed is None
        assert config.stop_sequences == []

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = GenerationConfig(
            temperature=0.3,
            max_tokens=2048,
            json_mode=False,
            seed=42,
        )
        assert config.temperature == 0.3
        assert config.max_tokens == 2048
        assert config.json_mode is False
        assert config.seed == 42


class TestGenerationResult:
    """Tests for GenerationResult."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating a generation result."""
        result = GenerationResult(
            content='{"id": "test"}',
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="gpt-4.1-mini",
        )
        assert result.content == '{"id": "test"}'
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 30
        assert result.model == "gpt-4.1-mini"
        assert result.raw_response is None


class TestOpenAIBackend:
    """Tests for OpenAI backend."""

    @pytest.mark.unit
    def test_requires_api_key(self):
        """Test that backend requires API key."""
        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            from .openai import OpenAIBackend

            with pytest.raises(AuthenticationError, match="API key required"):
                OpenAIBackend()
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original

    @pytest.mark.unit
    def test_creates_with_api_key(self):
        """Test backend creation with API key."""
        from .openai import OpenAIBackend

        backend = OpenAIBackend(api_key="test-key-12345")
        assert backend.provider == "openai"
        assert backend.model_name == "gpt-4.1-mini"  # default
        assert backend.supports_json_mode is True


class TestAnthropicBackend:
    """Tests for Anthropic backend."""

    @pytest.mark.unit
    def test_requires_api_key(self):
        """Test that backend requires API key."""
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from .anthropic import AnthropicBackend

            with pytest.raises(AuthenticationError, match="API key required"):
                AnthropicBackend()
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    @pytest.mark.unit
    def test_creates_with_api_key(self):
        """Test backend creation with API key."""
        from .anthropic import AnthropicBackend

        backend = AnthropicBackend(api_key="test-key-12345")
        assert backend.provider == "anthropic"
        assert backend.model_name == "claude-sonnet-4-5"  # default
        assert backend.supports_json_mode is False  # Anthropic uses prompt engineering


class TestDeepSeekBackend:
    """Tests for DeepSeek backend."""

    @pytest.mark.unit
    def test_requires_api_key(self):
        """Test that backend requires API key."""
        original = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            from .deepseek import DeepSeekBackend

            with pytest.raises(AuthenticationError, match="API key required"):
                DeepSeekBackend()
        finally:
            if original:
                os.environ["DEEPSEEK_API_KEY"] = original

    @pytest.mark.unit
    def test_creates_with_api_key(self):
        """Test backend creation with API key."""
        from .deepseek import DeepSeekBackend

        backend = DeepSeekBackend(api_key="test-key-12345")
        assert backend.provider == "deepseek"


class TestQwenBackend:
    """Tests for Qwen backend."""

    @pytest.mark.unit
    def test_requires_api_key(self):
        """Test that backend requires API key."""
        original = os.environ.pop("QWEN_API_KEY", None)
        try:
            from .qwen import QwenBackend

            with pytest.raises(AuthenticationError, match="API key required"):
                QwenBackend()
        finally:
            if original:
                os.environ["QWEN_API_KEY"] = original

    @pytest.mark.unit
    def test_creates_with_api_key(self):
        """Test backend creation with API key."""
        from .qwen import QwenBackend

        backend = QwenBackend(api_key="test-key-12345")
        assert backend.provider == "qwen"


class TestOllamaBackend:
    """Tests for Ollama backend."""

    @pytest.mark.unit
    def test_no_api_key_required(self):
        """Test that Ollama doesn't require API key."""
        from .ollama import OllamaBackend

        backend = OllamaBackend()
        assert backend.provider == "ollama"
        assert backend.model_name == "qwen3"  # default

    @pytest.mark.unit
    def test_custom_model(self):
        """Test custom model selection."""
        from .ollama import OllamaBackend

        backend = OllamaBackend(model="llama3.2")
        assert backend.model_name == "llama3.2"


class TestCreateLLMBackend:
    """Tests for create_llm_backend factory."""

    @pytest.mark.unit
    def test_creates_openai_backend(self):
        """Test factory creates OpenAI backend."""
        backend = create_llm_backend(LLMModel.GPT_4_1_MINI, api_key="test-key")
        assert backend.provider == "openai"

    @pytest.mark.unit
    def test_creates_anthropic_backend(self):
        """Test factory creates Anthropic backend."""
        backend = create_llm_backend(LLMModel.CLAUDE_SONNET_4_5, api_key="test-key")
        assert backend.provider == "anthropic"

    @pytest.mark.unit
    def test_creates_deepseek_backend(self):
        """Test factory creates DeepSeek backend."""
        backend = create_llm_backend(LLMModel.DEEPSEEK_V3_2, api_key="test-key")
        assert backend.provider == "deepseek"

    @pytest.mark.unit
    def test_creates_qwen_backend(self):
        """Test factory creates Qwen backend."""
        backend = create_llm_backend(LLMModel.QWEN_TURBO, api_key="test-key")
        assert backend.provider == "qwen"

    @pytest.mark.unit
    def test_creates_ollama_backend(self):
        """Test factory creates Ollama backend."""
        backend = create_llm_backend(LLMModel.OLLAMA_QWEN3)
        assert backend.provider == "ollama"

    @pytest.mark.unit
    def test_creates_from_string_name(self):
        """Test factory accepts string model names."""
        backend = create_llm_backend("gpt-4.1-mini", api_key="test-key")
        assert backend.model_name == "gpt-4.1-mini"
