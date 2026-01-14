"""LLM backend implementations.

Provides abstract base class and concrete implementations for
various LLM providers (OpenAI, Anthropic, DeepSeek, Qwen, Ollama).
"""

from .base import (
    AuthenticationError,
    ContextLengthError,
    GenerationConfig,
    GenerationResult,
    InvalidResponseError,
    LLMBackend,
    LLMError,
    RateLimitError,
)
from .factory import create_llm_backend
from .model_spec import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QWEN_MODEL,
    LLMCapability,
    LLMModel,
    LLMProviderType,
    LLMSpec,
    get_llm_spec,
)

__all__ = [
    # Base classes and types
    "LLMBackend",
    "GenerationConfig",
    "GenerationResult",
    # Exceptions
    "LLMError",
    "RateLimitError",
    "ContextLengthError",
    "InvalidResponseError",
    "AuthenticationError",
    # Model specification
    "LLMCapability",
    "LLMProviderType",
    "LLMSpec",
    "LLMModel",
    "get_llm_spec",
    # Defaults
    "DEFAULT_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_QWEN_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    # Factory
    "create_llm_backend",
]
