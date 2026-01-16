"""OpenAI GPT backend implementation.

Supports GPT-5.x, GPT-4.x, and other OpenAI models via the OpenAI API.
"""

import json
import logging
from typing import Any

from ...config import EnvVar, get_environment
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
from .model_spec import DEFAULT_OPENAI_MODEL, LLMCapability, get_llm_spec

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend.

    Uses OpenAI API for GPT-5.x, GPT-4.x, and other models.

    Environment:
        OPENAI_API_KEY: API key (required if not passed to constructor).

    Example:
        >>> backend = OpenAIBackend()
        >>> result = backend.generate("Generate a login form layout")
        >>> print(result.content)

        >>> backend = OpenAIBackend(model="gpt-5.1")
        >>> layout = backend.generate_json("Generate a login form as JSON")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL.spec.name,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize OpenAI backend.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model name (gpt-5.2, gpt-4.1-mini, etc.).
            base_url: Optional custom API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts for transient errors.

        Raises:
            AuthenticationError: If no API key available.
        """
        self._api_key = api_key or get_environment(EnvVar.OPENAI_API_KEY)
        if not self._api_key:
            raise AuthenticationError(
                "OpenAI API key required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._spec = get_llm_spec(model)
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize OpenAI client.

        Returns:
            OpenAI client instance.

        Raises:
            ImportError: If openai package not installed.
        """
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                )
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return self._spec.name

    @property
    def provider(self) -> str:
        """Get the provider identifier."""
        return "openai"

    @property
    def supports_json_mode(self) -> bool:
        """Check if backend natively supports JSON mode."""
        return self._spec.supports(LLMCapability.JSON_MODE)

    @property
    def context_window(self) -> int:
        """Get maximum context window size."""
        return self._spec.context_window

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text using OpenAI API.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system instruction.
            config: Generation configuration.

        Returns:
            GenerationResult with content and metadata.

        Raises:
            LLMError: If generation fails.
            RateLimitError: If rate limit exceeded.
            ContextLengthError: If prompt too long.
        """
        config = config or GenerationConfig()
        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._spec.name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        if config.json_mode and self.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        if config.seed is not None and self._spec.supports(LLMCapability.SEED):
            kwargs["seed"] = config.seed

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            self._handle_error(e)
            raise  # Re-raise if _handle_error doesn't raise

        choice = response.choices[0]
        return GenerationResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "unknown",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            raw_response=response,
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        """Generate and parse JSON response.

        Args:
            prompt: User prompt requesting JSON output.
            system_prompt: Optional system instruction.
            config: Generation configuration (json_mode forced True).

        Returns:
            Parsed JSON dictionary.

        Raises:
            InvalidResponseError: If response is not valid JSON.
        """
        config = config or GenerationConfig()
        json_config = GenerationConfig(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            json_mode=True,
            stop_sequences=config.stop_sequences,
            top_p=config.top_p,
            seed=config.seed,
        )

        result = self.generate(prompt, system_prompt=system_prompt, config=json_config)

        try:
            return json.loads(result.content)
        except json.JSONDecodeError as e:
            raise InvalidResponseError(
                f"Failed to parse JSON response: {e}\nContent: {result.content[:500]}"
            ) from e

    def _handle_error(self, error: Exception) -> None:
        """Convert provider errors to standard exceptions.

        Args:
            error: The caught exception.

        Raises:
            RateLimitError: For rate limit errors.
            ContextLengthError: For context length errors.
            AuthenticationError: For auth errors.
            LLMError: For other errors.
        """
        error_str = str(error).lower()

        if "rate limit" in error_str or "rate_limit" in error_str:
            raise RateLimitError(str(error))
        elif "context length" in error_str or "maximum context" in error_str:
            raise ContextLengthError(str(error))
        elif "authentication" in error_str or "invalid api key" in error_str:
            raise AuthenticationError(str(error))
        else:
            raise LLMError(str(error)) from error


__all__ = ["OpenAIBackend"]
