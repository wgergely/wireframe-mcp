"""Ollama local backend implementation.

Supports local LLM inference via Ollama server.
Models include DeepSeek-R1, Qwen3, Llama 3.2, Gemma 3, and more.
"""

import json
import logging
from typing import Any

from .base import (
    GenerationConfig,
    GenerationResult,
    InvalidResponseError,
    LLMBackend,
    LLMError,
)
from .model_spec import DEFAULT_OLLAMA_MODEL, LLMCapability, get_llm_spec

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Ollama local inference backend.

    Uses local Ollama server for LLM inference. No API key required.
    Ollama must be running locally with the desired model pulled.

    Environment:
        No API key required. Ollama server must be running.

    Example:
        >>> backend = OllamaBackend()
        >>> result = backend.generate("Generate a login form layout")
        >>> print(result.content)

        >>> backend = OllamaBackend(model="deepseek-r1")
        >>> layout = backend.generate_json("Generate a login form as JSON")

    Setup:
        1. Install Ollama: https://ollama.com
        2. Pull a model: `ollama pull qwen3`
        3. Ollama server runs automatically on localhost:11434
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL.spec.name,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        """Initialize Ollama backend.

        Args:
            model: Model name (qwen3, deepseek-r1, llama3.2, etc.).
            base_url: Ollama server URL.
            timeout: Request timeout in seconds (local inference can be slow).
        """
        self._spec = get_llm_spec(model)
        self._base_url = base_url
        self._timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize Ollama client.

        Returns:
            Ollama client instance.

        Raises:
            ImportError: If ollama package not installed.
        """
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self._base_url)
            except ImportError as e:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return self._spec.name

    @property
    def provider(self) -> str:
        """Get the provider identifier."""
        return "ollama"

    @property
    def supports_json_mode(self) -> bool:
        """Check if backend natively supports JSON mode.

        Ollama supports JSON format via the format parameter.
        """
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
        """Generate text using Ollama.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system instruction.
            config: Generation configuration.

        Returns:
            GenerationResult with content and metadata.

        Raises:
            LLMError: If generation fails (e.g., model not found, server down).
        """
        config = config or GenerationConfig()
        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
            "top_p": config.top_p,
        }

        if config.seed is not None:
            options["seed"] = config.seed

        kwargs: dict[str, Any] = {
            "model": self._spec.name,
            "messages": messages,
            "options": options,
        }

        if config.json_mode and self.supports_json_mode:
            kwargs["format"] = "json"

        try:
            response = client.chat(**kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "pull" in error_msg:
                raise LLMError(
                    f"Model '{self._spec.name}' not found. "
                    f"Pull it first with: ollama pull {self._spec.name}"
                ) from e
            elif "connection" in error_msg or "refused" in error_msg:
                raise LLMError(
                    "Cannot connect to Ollama server. "
                    "Ensure Ollama is running: https://ollama.com"
                ) from e
            else:
                raise LLMError(str(e)) from e

        content = response.get("message", {}).get("content", "")

        return GenerationResult(
            content=content,
            finish_reason="stop",
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": (
                    response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                ),
            },
            model=self._spec.name,
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


__all__ = ["OllamaBackend"]
