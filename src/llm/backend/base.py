"""Abstract base class for LLM backends.

Defines the interface that all LLM provider implementations must follow.
Follows the same patterns as EmbeddingBackend in src/vector/backend/base.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationConfig:
    """Configuration for LLM text generation.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
        max_tokens: Maximum tokens to generate in response.
        json_mode: Whether to enforce JSON output format.
        stop_sequences: Optional sequences that stop generation.
        top_p: Nucleus sampling parameter (0.0-1.0).
        seed: Optional seed for reproducible generation.
    """

    temperature: float = 0.7
    max_tokens: int = 4096
    json_mode: bool = True
    stop_sequences: list[str] = field(default_factory=list)
    top_p: float = 1.0
    seed: int | None = None


@dataclass
class GenerationResult:
    """Result from LLM text generation.

    Attributes:
        content: Generated text content.
        finish_reason: Why generation stopped ('stop', 'length', 'content_filter').
        usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens).
        model: Model identifier that was used.
        raw_response: Provider-specific raw response for debugging.
    """

    content: str
    finish_reason: str
    usage: dict[str, int]
    model: str
    raw_response: Any = None


class LLMBackend(ABC):
    """Abstract interface for LLM text generation backends.

    LLM backends convert prompts into structured text responses.
    Implementations may use external APIs (OpenAI, Anthropic, DeepSeek, Qwen)
    or local models (Ollama).

    Example:
        >>> backend = OpenAIBackend(model="gpt-4.1-mini")
        >>> result = backend.generate("Generate a login form layout")
        >>> print(result.content)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system instruction for context.
            config: Generation configuration options.

        Returns:
            GenerationResult with generated content and metadata.

        Raises:
            LLMError: If generation fails.
            RateLimitError: If API rate limit is exceeded.
            ContextLengthError: If prompt exceeds context window.
        """

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        """Generate and parse JSON response.

        Convenience method that enables JSON mode and parses output.

        Args:
            prompt: User prompt requesting JSON output.
            system_prompt: Optional system instruction.
            config: Generation configuration (json_mode forced True).

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMError: If generation fails.
            InvalidResponseError: If response is not valid JSON.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model identifier.

        Returns:
            String model name (e.g., 'gpt-4.1-mini', 'claude-sonnet-4-5').
        """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider identifier.

        Returns:
            String provider name (e.g., 'openai', 'anthropic').
        """

    @property
    def name(self) -> str:
        """Get backend identifier for logging.

        Returns:
            String in format 'provider:model'.
        """
        return f"{self.provider}:{self.model_name}"

    @property
    @abstractmethod
    def supports_json_mode(self) -> bool:
        """Check if backend natively supports JSON mode.

        Returns:
            True if native JSON mode available.
        """

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Get maximum context window size in tokens.

        Returns:
            Maximum tokens in context window.
        """


class LLMError(Exception):
    """Base exception for LLM backend errors."""


class RateLimitError(LLMError):
    """Raised when API rate limit is exceeded.

    Attributes:
        retry_after: Suggested wait time in seconds before retry.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ContextLengthError(LLMError):
    """Raised when prompt exceeds the model's context window."""


class InvalidResponseError(LLMError):
    """Raised when response cannot be parsed as expected format."""


class AuthenticationError(LLMError):
    """Raised when API authentication fails (invalid or missing key)."""


__all__ = [
    "LLMBackend",
    "GenerationConfig",
    "GenerationResult",
    "LLMError",
    "RateLimitError",
    "ContextLengthError",
    "InvalidResponseError",
    "AuthenticationError",
]
