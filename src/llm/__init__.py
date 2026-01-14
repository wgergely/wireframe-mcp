"""LLM integration layer for wireframe generation.

This module provides multi-provider LLM support for generating
UI layouts from natural language descriptions.

Main components:
- LayoutGenerator: Orchestrates prompt building, LLM calls, and validation
- LLMBackend: Abstract interface for LLM providers
- create_llm_backend: Factory function for creating backends

Supported providers:
- OpenAI (GPT-5.x, GPT-4.x)
- Anthropic (Claude 4.5, Claude 4.1)
- DeepSeek (V3.2, R1, Coder)
- Qwen (Qwen3-Max, Turbo, Plus)
- Ollama (local models)

Example:
    >>> from src.llm import LayoutGenerator
    >>> generator = LayoutGenerator()
    >>> output = generator.generate("login form with email and password")
    >>> print(output.context.node)

    >>> # With specific model
    >>> from src.llm import create_llm_backend, LLMModel
    >>> backend = create_llm_backend(LLMModel.CLAUDE_SONNET_4_5)
    >>> generator = LayoutGenerator(backend=backend)
"""

from .backend import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QWEN_MODEL,
    AuthenticationError,
    ContextLengthError,
    GenerationConfig,
    GenerationResult,
    InvalidResponseError,
    LLMBackend,
    LLMCapability,
    LLMError,
    LLMModel,
    LLMProviderType,
    LLMSpec,
    RateLimitError,
    create_llm_backend,
    get_llm_spec,
)
from .generator import (
    GenerationOutput,
    GenerationStats,
    GeneratorConfig,
    LayoutGenerator,
    RetryConfig,
    RetryStrategy,
)

__all__ = [
    # Main API
    "LayoutGenerator",
    "create_llm_backend",
    # Generator types
    "GeneratorConfig",
    "GenerationStats",
    "GenerationOutput",
    "RetryConfig",
    "RetryStrategy",
    # Backend types
    "LLMBackend",
    "GenerationConfig",
    "GenerationResult",
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
    # Exceptions
    "LLMError",
    "RateLimitError",
    "ContextLengthError",
    "InvalidResponseError",
    "AuthenticationError",
]
