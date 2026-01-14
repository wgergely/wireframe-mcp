"""Qwen backend implementation (OpenAI-compatible API).

Supports Qwen3-Max, Qwen-Turbo, Qwen-Plus, and other Alibaba Qwen models.
"""

import os

from .base import AuthenticationError
from .model_spec import DEFAULT_QWEN_MODEL, get_llm_spec
from .openai import OpenAIBackend


class QwenBackend(OpenAIBackend):
    """Qwen backend using OpenAI-compatible API.

    Qwen (Alibaba) provides an OpenAI-compatible API via DashScope,
    so this backend extends OpenAIBackend with the appropriate configuration.

    Environment:
        QWEN_API_KEY: API key (required if not passed to constructor).

    Example:
        >>> backend = QwenBackend()
        >>> result = backend.generate("Generate a login form layout")
        >>> print(result.content)

        >>> backend = QwenBackend(model="qwen3-max")
        >>> layout = backend.generate_json("Generate a login form as JSON")
    """

    QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_QWEN_MODEL.spec.name,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize Qwen backend.

        Args:
            api_key: Qwen/DashScope API key. Falls back to QWEN_API_KEY env var.
            model: Model name (qwen3-max, qwen-turbo, etc.).
            base_url: Optional custom API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts for transient errors.

        Raises:
            AuthenticationError: If no API key available.
        """
        resolved_api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not resolved_api_key:
            raise AuthenticationError(
                "Qwen API key required. Set QWEN_API_KEY environment "
                "variable or pass api_key parameter."
            )

        # Get spec to validate model exists
        spec = get_llm_spec(model)

        # Use Qwen base URL
        resolved_base_url = base_url or spec.base_url or self.QWEN_BASE_URL

        # Initialize parent with resolved values
        self._api_key = resolved_api_key
        self._spec = spec
        self._base_url = resolved_base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None

    @property
    def provider(self) -> str:
        """Get the provider identifier."""
        return "qwen"


__all__ = ["QwenBackend"]
