"""Voyage AI embedding backend.

Uses Voyage AI API for high-quality embeddings with large context window.
Supports batch processing with automatic rate limiting.
"""

import logging
import os
import time
from typing import Any

import numpy as np

from .base import EmbeddingBackend

logger = logging.getLogger(__name__)

# Voyage model dimensions
VOYAGE_DIMENSIONS = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-large-2": 1536,
    "voyage-2": 1024,
}

# Default batch size for API calls
DEFAULT_BATCH_SIZE = 128

# Rate limit delay between batches (seconds)
RATE_LIMIT_DELAY = 0.1


class VoyageBackend(EmbeddingBackend):
    """Voyage AI embedding backend.

    Uses Voyage AI's embedding API for high-quality vector generation.
    Supports large context windows suitable for full LayoutNode trees.

    Environment:
        VOYAGE_API_KEY: API key for Voyage AI (required if not passed).

    Example:
        >>> backend = VoyageBackend()
        >>> vectors = backend.embed(["login form", "dashboard"])
        >>> print(vectors.shape)
        (2, 1024)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
    ):
        """Initialize Voyage backend.

        Args:
            api_key: Voyage API key. Falls back to VOYAGE_API_KEY env var.
            model: Voyage model name (voyage-3, voyage-3-lite, etc.).
            batch_size: Maximum texts per API call.
            max_retries: Number of retry attempts on failure.

        Raises:
            ValueError: If no API key available.
        """
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._client: Any = None

        # Validate model
        if model not in VOYAGE_DIMENSIONS:
            logger.warning(
                f"Unknown model '{model}', assuming dimension 1024. "
                f"Known models: {list(VOYAGE_DIMENSIONS.keys())}"
            )

    @property
    def dimension(self) -> int:
        """Get embedding dimension for configured model."""
        return VOYAGE_DIMENSIONS.get(self._model, 1024)

    @property
    def name(self) -> str:
        """Get backend identifier."""
        return f"voyage:{self._model}"

    def _get_client(self) -> Any:
        """Lazily initialize Voyage client.

        Returns:
            Initialized voyageai.Client instance.

        Raises:
            ImportError: If voyageai package not installed.
        """
        if self._client is None:
            try:
                import voyageai

                self._client = voyageai.Client(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "voyageai package required for VoyageBackend. "
                    "Install with: pip install voyageai"
                ) from e
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            NumPy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        client = self._get_client()
        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._embed_batch_with_retry(client, batch)
            all_embeddings.extend(embeddings)

            # Rate limiting between batches
            if i + self._batch_size < len(texts):
                time.sleep(RATE_LIMIT_DELAY)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for single query.

        Voyage uses same model for queries and documents by default.

        Args:
            query: Query text to embed.

        Returns:
            NumPy array of shape (dimension,).
        """
        result = self.embed([query])
        return result[0]

    def _embed_batch_with_retry(
        self, client: Any, texts: list[str]
    ) -> list[list[float]]:
        """Embed batch with retry logic.

        Args:
            client: Voyage client instance.
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            Exception: If all retries exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                result = client.embed(texts, model=self._model)
                return result.embeddings
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Voyage API error (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    time.sleep(2**attempt)

        raise last_error or RuntimeError("Unknown error in Voyage API")


__all__ = ["VoyageBackend"]
