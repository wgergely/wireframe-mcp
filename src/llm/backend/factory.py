"""Backend factory for creating LLM backends from model specifications.

Provides a unified entry point for creating any supported LLM backend.
"""

from .base import LLMBackend
from .model_spec import (
    DEFAULT_MODEL,
    LLMModel,
    LLMProviderType,
    LLMSpec,
    get_llm_spec,
)


def create_llm_backend(
    model: str | LLMModel | LLMSpec = DEFAULT_MODEL,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> LLMBackend:
    """Create an LLM backend from a model specification.

    This is the primary factory function for instantiating LLM backends.
    It automatically routes to the appropriate backend class based on
    the model's provider.

    Args:
        model: Model to use. Can be:
            - String model name (e.g., "gpt-4.1-mini", "claude-sonnet-4-5")
            - LLMModel enum value (e.g., LLMModel.GPT_4_1_MINI)
            - LLMSpec instance
        api_key: API key for remote providers. Falls back to environment
            variable if not provided.
        base_url: Optional custom API endpoint. Uses provider default if None.
        **kwargs: Additional arguments passed to backend constructor
            (e.g., timeout, max_retries).

    Returns:
        Configured LLMBackend instance.

    Raises:
        ValueError: If model is unknown or configuration invalid.
        AuthenticationError: If API key required but not provided.

    Example:
        >>> # Using default model (gpt-4.1-mini)
        >>> backend = create_llm_backend()

        >>> # Using model name string
        >>> backend = create_llm_backend("claude-sonnet-4-5")

        >>> # Using LLMModel enum
        >>> backend = create_llm_backend(LLMModel.DEEPSEEK_V3_2)

        >>> # With custom configuration
        >>> backend = create_llm_backend(
        ...     "gpt-4.1-mini",
        ...     api_key="sk-...",
        ...     timeout=120.0,
        ... )
    """
    spec = get_llm_spec(model)

    if spec.provider == LLMProviderType.OPENAI:
        from .openai import OpenAIBackend

        return OpenAIBackend(
            api_key=api_key,
            model=spec.name,
            base_url=base_url,
            **kwargs,
        )

    if spec.provider == LLMProviderType.ANTHROPIC:
        from .anthropic import AnthropicBackend

        return AnthropicBackend(
            api_key=api_key,
            model=spec.name,
            **kwargs,
        )

    if spec.provider == LLMProviderType.DEEPSEEK:
        from .deepseek import DeepSeekBackend

        return DeepSeekBackend(
            api_key=api_key,
            model=spec.name,
            base_url=base_url,
            **kwargs,
        )

    if spec.provider == LLMProviderType.QWEN:
        from .qwen import QwenBackend

        return QwenBackend(
            api_key=api_key,
            model=spec.name,
            base_url=base_url,
            **kwargs,
        )

    if spec.provider == LLMProviderType.OLLAMA:
        from .ollama import OllamaBackend

        return OllamaBackend(
            model=spec.name,
            base_url=base_url or spec.base_url or "http://localhost:11434",
            **kwargs,
        )

    raise ValueError(f"Unsupported provider type: {spec.provider}")


__all__ = ["create_llm_backend"]
