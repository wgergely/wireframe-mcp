"""Anthropic Claude backend implementation.

Supports Claude 4.5, Claude 4.1, and other Claude models via the Anthropic API.
"""

import json
import logging
import os
import re
from typing import Any

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
from .model_spec import DEFAULT_ANTHROPIC_MODEL, get_llm_spec

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend.

    Uses Anthropic API for Claude 4.5, Claude 4.1, and other Claude models.

    Note: Anthropic does not have native JSON mode. This backend uses
    prompt engineering and response extraction to handle JSON requests.

    Environment:
        ANTHROPIC_API_KEY: API key (required if not passed to constructor).

    Example:
        >>> backend = AnthropicBackend()
        >>> result = backend.generate("Generate a login form layout")
        >>> print(result.content)

        >>> backend = AnthropicBackend(model="claude-sonnet-4-5")
        >>> layout = backend.generate_json("Generate a login form as JSON")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_ANTHROPIC_MODEL.spec.name,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize Anthropic backend.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Model name (claude-sonnet-4-5, claude-opus-4-5, etc.).
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts for transient errors.

        Raises:
            AuthenticationError: If no API key available.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._spec = get_llm_spec(model)
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize Anthropic client.

        Returns:
            Anthropic client instance.

        Raises:
            ImportError: If anthropic package not installed.
        """
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                )
            except ImportError as e:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return self._spec.name

    @property
    def provider(self) -> str:
        """Get the provider identifier."""
        return "anthropic"

    @property
    def supports_json_mode(self) -> bool:
        """Check if backend natively supports JSON mode.

        Anthropic does not have native JSON mode - uses prompt engineering.
        """
        return False

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
        """Generate text using Anthropic API.

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

        # For JSON mode, enhance the prompt
        effective_prompt = prompt
        if config.json_mode:
            effective_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT: Respond with valid JSON only. "
                "Do not include any text, explanation, or markdown formatting "
                "before or after the JSON object."
            )

        kwargs: dict[str, Any] = {
            "model": self._spec.name,
            "messages": [{"role": "user", "content": effective_prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        try:
            response = client.messages.create(**kwargs)
        except Exception as e:
            self._handle_error(e)
            raise

        content = ""
        if response.content:
            content = response.content[0].text

        return GenerationResult(
            content=content,
            finish_reason=response.stop_reason or "unknown",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": (
                    response.usage.input_tokens + response.usage.output_tokens
                ),
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

        Since Anthropic doesn't have native JSON mode, this method:
        1. Adds JSON instructions to the prompt
        2. Extracts JSON from markdown code blocks if present
        3. Parses the result

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

        # Extract JSON from the response
        content = self._extract_json(result.content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise InvalidResponseError(
                f"Failed to parse JSON response: {e}\nContent: {result.content[:500]}"
            ) from e

    def _extract_json(self, content: str) -> str:
        """Extract JSON from response, handling markdown code blocks.

        Args:
            content: Raw response content.

        Returns:
            Extracted JSON string.
        """
        content = content.strip()

        # Try to extract from markdown code blocks
        # Match ```json ... ``` or ``` ... ```
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, content)
        if matches:
            # Use the first match that looks like JSON
            for match in matches:
                stripped = match.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    return stripped

        # Remove any leading/trailing markdown if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

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
        elif "context length" in error_str or "too long" in error_str:
            raise ContextLengthError(str(error))
        elif "authentication" in error_str or "invalid" in error_str:
            raise AuthenticationError(str(error))
        else:
            raise LLMError(str(error)) from error


__all__ = ["AnthropicBackend"]
