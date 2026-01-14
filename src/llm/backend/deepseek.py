"""DeepSeek backend implementation (OpenAI-compatible API).

Supports DeepSeek V3.2, DeepSeek R1, and DeepSeek Coder models.
"""

import os

from .base import AuthenticationError
from .model_spec import DEFAULT_DEEPSEEK_MODEL, get_llm_spec
from .openai import OpenAIBackend


class DeepSeekBackend(OpenAIBackend):
    """DeepSeek backend using OpenAI-compatible API.

    DeepSeek provides an OpenAI-compatible API endpoint, so this backend
    extends OpenAIBackend with the appropriate base URL and API key handling.

    Environment:
        DEEPSEEK_API_KEY: API key (required if not passed to constructor).

    Example:
        >>> backend = DeepSeekBackend()
        >>> result = backend.generate("Generate a login form layout")
        >>> print(result.content)

        >>> backend = DeepSeekBackend(model="deepseek-reasoner")
        >>> layout = backend.generate_json("Generate a login form as JSON")
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_DEEPSEEK_MODEL.spec.name,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize DeepSeek backend.

        Args:
            api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
            model: Model name (deepseek-chat, deepseek-reasoner, etc.).
            base_url: Optional custom API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts for transient errors.

        Raises:
            AuthenticationError: If no API key available.
        """
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_api_key:
            raise AuthenticationError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment "
                "variable or pass api_key parameter."
            )

        # Get spec to validate model exists
        spec = get_llm_spec(model)

        # Use DeepSeek base URL
        resolved_base_url = base_url or spec.base_url or self.DEEPSEEK_BASE_URL

        # Initialize parent with resolved values
        # We need to bypass the parent's API key check
        self._api_key = resolved_api_key
        self._spec = spec
        self._base_url = resolved_base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None

    @property
    def provider(self) -> str:
        """Get the provider identifier."""
        return "deepseek"


__all__ = ["DeepSeekBackend"]
