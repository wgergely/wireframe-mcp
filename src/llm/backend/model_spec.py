"""Model specification system for LLM backends.

Provides a registry of supported LLM models with their capabilities,
context windows, and provider information.

Updated: January 2026 - Research-verified model specifications.
"""

from dataclasses import dataclass, field
from enum import Enum


class LLMCapability(Enum):
    """Capabilities that an LLM model may support."""

    JSON_MODE = "json_mode"  # Native JSON output enforcement
    FUNCTION_CALLING = "function_calling"  # Tool/function calling support
    VISION = "vision"  # Image input support
    STREAMING = "streaming"  # Streaming response support
    SYSTEM_PROMPT = "system_prompt"  # Dedicated system role
    SEED = "seed"  # Reproducible generation with seed
    EXTENDED_THINKING = "extended_thinking"  # Extended reasoning mode


class LLMProviderType(Enum):
    """Available LLM backend providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class LLMSpec:
    """Specification for an LLM model.

    Attributes:
        name: Model identifier (e.g., 'gpt-4.1-mini', 'claude-sonnet-4-5').
        provider: Backend provider type.
        context_window: Maximum context size in tokens.
        max_output_tokens: Maximum generation tokens.
        capabilities: Set of supported capabilities.
        description: Human-readable description.
        requires_api_key: Whether this model needs an API key.
        api_key_env_var: Environment variable name for API key.
        base_url: Optional custom API endpoint.
        cost_per_1m_input: Cost per 1M input tokens (USD).
        cost_per_1m_output: Cost per 1M output tokens (USD).
    """

    name: str
    provider: LLMProviderType
    context_window: int
    max_output_tokens: int
    capabilities: frozenset[LLMCapability] = field(default_factory=frozenset)
    description: str = ""
    requires_api_key: bool = True
    api_key_env_var: str = ""
    base_url: str | None = None
    cost_per_1m_input: float | None = None
    cost_per_1m_output: float | None = None

    def supports(self, capability: LLMCapability) -> bool:
        """Check if model supports a capability."""
        return capability in self.capabilities

    @property
    def is_local(self) -> bool:
        """Check if model runs locally."""
        return self.provider == LLMProviderType.OLLAMA

    @property
    def is_remote(self) -> bool:
        """Check if model uses remote API."""
        return not self.is_local


# Common capability sets
_OPENAI_FULL = frozenset(
    {
        LLMCapability.JSON_MODE,
        LLMCapability.FUNCTION_CALLING,
        LLMCapability.VISION,
        LLMCapability.STREAMING,
        LLMCapability.SYSTEM_PROMPT,
        LLMCapability.SEED,
    }
)

_ANTHROPIC_FULL = frozenset(
    {
        LLMCapability.FUNCTION_CALLING,
        LLMCapability.VISION,
        LLMCapability.STREAMING,
        LLMCapability.SYSTEM_PROMPT,
        LLMCapability.EXTENDED_THINKING,
    }
)

_DEEPSEEK_FULL = frozenset(
    {
        LLMCapability.JSON_MODE,
        LLMCapability.FUNCTION_CALLING,
        LLMCapability.STREAMING,
        LLMCapability.SYSTEM_PROMPT,
    }
)

_QWEN_FULL = frozenset(
    {
        LLMCapability.JSON_MODE,
        LLMCapability.FUNCTION_CALLING,
        LLMCapability.STREAMING,
        LLMCapability.SYSTEM_PROMPT,
    }
)

_OLLAMA_BASIC = frozenset(
    {
        LLMCapability.JSON_MODE,
        LLMCapability.STREAMING,
        LLMCapability.SYSTEM_PROMPT,
    }
)


class LLMModel(Enum):
    """Registry of available LLM models (January 2026)."""

    # === OpenAI Models ===
    GPT_5_2 = LLMSpec(
        name="gpt-5.2",
        provider=LLMProviderType.OPENAI,
        context_window=128000,
        max_output_tokens=16384,
        capabilities=_OPENAI_FULL,
        description="OpenAI flagship for professional work and agents",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
    )

    GPT_5_1 = LLMSpec(
        name="gpt-5.1",
        provider=LLMProviderType.OPENAI,
        context_window=128000,
        max_output_tokens=16384,
        capabilities=_OPENAI_FULL,
        description="OpenAI balanced intelligence and speed",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
    )

    GPT_4_1 = LLMSpec(
        name="gpt-4.1",
        provider=LLMProviderType.OPENAI,
        context_window=128000,
        max_output_tokens=16384,
        capabilities=_OPENAI_FULL,
        description="OpenAI developer favorite for coding",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
    )

    GPT_4_1_MINI = LLMSpec(
        name="gpt-4.1-mini",
        provider=LLMProviderType.OPENAI,
        context_window=128000,
        max_output_tokens=16384,
        capabilities=_OPENAI_FULL,
        description="OpenAI fast and efficient small model",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
    )

    GPT_4_5 = LLMSpec(
        name="gpt-4.5",
        provider=LLMProviderType.OPENAI,
        context_window=128000,
        max_output_tokens=16384,
        capabilities=_OPENAI_FULL,
        description="OpenAI high EQ for creative tasks",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
    )

    # === Anthropic Claude Models ===
    CLAUDE_OPUS_4_5 = LLMSpec(
        name="claude-opus-4-5",
        provider=LLMProviderType.ANTHROPIC,
        context_window=200000,
        max_output_tokens=64000,
        capabilities=_ANTHROPIC_FULL,
        description="Anthropic most intelligent model",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        cost_per_1m_input=5.0,
        cost_per_1m_output=25.0,
    )

    CLAUDE_SONNET_4_5 = LLMSpec(
        name="claude-sonnet-4-5",
        provider=LLMProviderType.ANTHROPIC,
        context_window=200000,
        max_output_tokens=64000,
        capabilities=_ANTHROPIC_FULL,
        description="Anthropic best balanced for coding and agents",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
    )

    CLAUDE_HAIKU_4_5 = LLMSpec(
        name="claude-haiku-4-5",
        provider=LLMProviderType.ANTHROPIC,
        context_window=200000,
        max_output_tokens=64000,
        capabilities=_ANTHROPIC_FULL,
        description="Anthropic fastest model",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        cost_per_1m_input=1.0,
        cost_per_1m_output=5.0,
    )

    CLAUDE_OPUS_4_1 = LLMSpec(
        name="claude-opus-4-1",
        provider=LLMProviderType.ANTHROPIC,
        context_window=200000,
        max_output_tokens=32000,
        capabilities=_ANTHROPIC_FULL,
        description="Anthropic legacy Opus (4.1)",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        cost_per_1m_input=15.0,
        cost_per_1m_output=75.0,
    )

    CLAUDE_SONNET_4 = LLMSpec(
        name="claude-sonnet-4-0",
        provider=LLMProviderType.ANTHROPIC,
        context_window=200000,
        max_output_tokens=64000,
        capabilities=_ANTHROPIC_FULL,
        description="Anthropic legacy Sonnet (4.0)",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
    )

    # === DeepSeek Models ===
    DEEPSEEK_V3_2 = LLMSpec(
        name="deepseek-chat",
        provider=LLMProviderType.DEEPSEEK,
        context_window=64000,
        max_output_tokens=8192,
        capabilities=_DEEPSEEK_FULL,
        description="DeepSeek V3.2 main model, GPT-5 level",
        requires_api_key=True,
        api_key_env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        cost_per_1m_input=0.14,
        cost_per_1m_output=0.28,
    )

    DEEPSEEK_R1 = LLMSpec(
        name="deepseek-reasoner",
        provider=LLMProviderType.DEEPSEEK,
        context_window=64000,
        max_output_tokens=8192,
        capabilities=_DEEPSEEK_FULL,
        description="DeepSeek R1 reasoning model",
        requires_api_key=True,
        api_key_env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        cost_per_1m_input=0.55,
        cost_per_1m_output=2.19,
    )

    DEEPSEEK_CODER = LLMSpec(
        name="deepseek-coder",
        provider=LLMProviderType.DEEPSEEK,
        context_window=64000,
        max_output_tokens=8192,
        capabilities=_DEEPSEEK_FULL,
        description="DeepSeek code specialized model",
        requires_api_key=True,
        api_key_env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        cost_per_1m_input=0.14,
        cost_per_1m_output=0.28,
    )

    # === Qwen Models (Alibaba) ===
    QWEN3_MAX = LLMSpec(
        name="qwen3-max",
        provider=LLMProviderType.QWEN,
        context_window=131072,
        max_output_tokens=8192,
        capabilities=_QWEN_FULL,
        description="Qwen3 flagship, SOTA for agents",
        requires_api_key=True,
        api_key_env_var="QWEN_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    QWEN3_CODER_PLUS = LLMSpec(
        name="qwen3-coder-plus",
        provider=LLMProviderType.QWEN,
        context_window=131072,
        max_output_tokens=8192,
        capabilities=_QWEN_FULL,
        description="Qwen3 code specialized with cache",
        requires_api_key=True,
        api_key_env_var="QWEN_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    QWEN_TURBO = LLMSpec(
        name="qwen-turbo",
        provider=LLMProviderType.QWEN,
        context_window=131072,
        max_output_tokens=8192,
        capabilities=_QWEN_FULL,
        description="Qwen fast inference model",
        requires_api_key=True,
        api_key_env_var="QWEN_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    QWEN_PLUS = LLMSpec(
        name="qwen-plus",
        provider=LLMProviderType.QWEN,
        context_window=131072,
        max_output_tokens=8192,
        capabilities=_QWEN_FULL,
        description="Qwen balanced general purpose",
        requires_api_key=True,
        api_key_env_var="QWEN_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    # === Ollama Local Models ===
    OLLAMA_DEEPSEEK_R1 = LLMSpec(
        name="deepseek-r1",
        provider=LLMProviderType.OLLAMA,
        context_window=64000,
        max_output_tokens=8192,
        capabilities=_OLLAMA_BASIC,
        description="DeepSeek R1 via Ollama (local)",
        requires_api_key=False,
        base_url="http://localhost:11434",
    )

    OLLAMA_QWEN3 = LLMSpec(
        name="qwen3",
        provider=LLMProviderType.OLLAMA,
        context_window=32768,
        max_output_tokens=8192,
        capabilities=_OLLAMA_BASIC,
        description="Qwen3 via Ollama (local)",
        requires_api_key=False,
        base_url="http://localhost:11434",
    )

    OLLAMA_LLAMA3_2 = LLMSpec(
        name="llama3.2",
        provider=LLMProviderType.OLLAMA,
        context_window=128000,
        max_output_tokens=4096,
        capabilities=_OLLAMA_BASIC,
        description="Meta Llama 3.2 via Ollama (local)",
        requires_api_key=False,
        base_url="http://localhost:11434",
    )

    OLLAMA_GEMMA3 = LLMSpec(
        name="gemma3",
        provider=LLMProviderType.OLLAMA,
        context_window=32768,
        max_output_tokens=8192,
        capabilities=_OLLAMA_BASIC,
        description="Google Gemma 3 via Ollama (local)",
        requires_api_key=False,
        base_url="http://localhost:11434",
    )

    OLLAMA_DOLPHIN = LLMSpec(
        name="dolphin3",
        provider=LLMProviderType.OLLAMA,
        context_window=32768,
        max_output_tokens=8192,
        capabilities=_OLLAMA_BASIC,
        description="Dolphin 3.0 via Ollama (local)",
        requires_api_key=False,
        base_url="http://localhost:11434",
    )

    @property
    def spec(self) -> LLMSpec:
        """Get the LLMSpec for this model."""
        return self.value

    @classmethod
    def by_name(cls, name: str) -> "LLMModel | None":
        """Look up model by name string.

        Args:
            name: Model name to find.

        Returns:
            LLMModel if found, None otherwise.
        """
        for model in cls:
            if model.spec.name == name:
                return model
        return None

    @classmethod
    def list_by_provider(cls, provider: LLMProviderType) -> list["LLMModel"]:
        """Get all models for a specific provider.

        Args:
            provider: Provider to filter by.

        Returns:
            List of LLMModel values for that provider.
        """
        return [m for m in cls if m.spec.provider == provider]

    @classmethod
    def list_local(cls) -> list["LLMModel"]:
        """Get all local (Ollama) models."""
        return cls.list_by_provider(LLMProviderType.OLLAMA)

    @classmethod
    def list_remote(cls) -> list["LLMModel"]:
        """Get all remote API models."""
        return [m for m in cls if m.spec.is_remote]


# Default models for each provider
DEFAULT_OPENAI_MODEL = LLMModel.GPT_4_1_MINI
DEFAULT_ANTHROPIC_MODEL = LLMModel.CLAUDE_SONNET_4_5
DEFAULT_DEEPSEEK_MODEL = LLMModel.DEEPSEEK_V3_2
DEFAULT_QWEN_MODEL = LLMModel.QWEN_TURBO
DEFAULT_OLLAMA_MODEL = LLMModel.OLLAMA_LLAMA3_2  # Widely available

# Overall default
DEFAULT_MODEL = DEFAULT_OPENAI_MODEL


def get_llm_spec(model: str | LLMModel | LLMSpec) -> LLMSpec:
    """Resolve a model reference to its LLMSpec.

    Args:
        model: Can be a model name string, LLMModel enum, or LLMSpec.

    Returns:
        The resolved LLMSpec.

    Raises:
        ValueError: If model name is not found.
    """
    if isinstance(model, LLMSpec):
        return model
    if isinstance(model, LLMModel):
        return model.spec
    found = LLMModel.by_name(model)
    if found:
        return found.spec
    raise ValueError(f"Unknown model: {model}")


__all__ = [
    "LLMCapability",
    "LLMProviderType",
    "LLMSpec",
    "LLMModel",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_QWEN_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_MODEL",
    "get_llm_spec",
]
