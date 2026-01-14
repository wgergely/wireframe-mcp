"""Voyage AI embedding backend.

Uses Voyage AI API for high-quality embeddings with large context window.
Supports batch processing with adaptive rate limiting and concurrent requests.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..types import (
    VOYAGE_BATCH_SIZE,
    VOYAGE_CONCURRENT_BATCHES,
    VOYAGE_RATE_LIMIT_FAST_THRESHOLD,
    VOYAGE_RATE_LIMIT_MIN_DELAY,
    VOYAGE_RATE_LIMIT_MODERATE_DELAY,
    VOYAGE_RATE_LIMIT_MODERATE_THRESHOLD,
    VOYAGE_RETRY_BASE,
)
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


class VoyageBackend(EmbeddingBackend):
    """Voyage AI embedding backend.

    Uses Voyage AI's embedding API for high-quality vector generation.
    Supports large context windows suitable for full LayoutNode trees.

    Features:
        - Adaptive rate limiting based on API response times
        - Concurrent batch processing for faster throughput
        - Automatic retry with exponential backoff

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
        batch_size: int = VOYAGE_BATCH_SIZE,
        max_retries: int = 3,
        concurrent_batches: int = VOYAGE_CONCURRENT_BATCHES,
        adaptive_rate_limit: bool = True,
    ):
        """Initialize Voyage backend.

        Args:
            api_key: Voyage API key. Falls back to VOYAGE_API_KEY env var.
            model: Voyage model name (voyage-3, voyage-3-lite, etc.).
            batch_size: Maximum texts per API call.
            max_retries: Number of retry attempts on failure.
            concurrent_batches: Number of concurrent API requests.
            adaptive_rate_limit: Use adaptive rate limiting based on response time.

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
        self._concurrent_batches = concurrent_batches
        self._adaptive_rate_limit = adaptive_rate_limit
        self._client: Any = None
        self._last_response_time: float = 0.0

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

        Uses concurrent processing and adaptive rate limiting for optimal
        throughput while respecting API limits.

        Args:
            texts: List of text strings to embed.

        Returns:
            NumPy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        client = self._get_client()

        # Split texts into batches
        batches: list[tuple[int, list[str]]] = []
        for i in range(0, len(texts), self._batch_size):
            batches.append((i, texts[i : i + self._batch_size]))

        # Use concurrent processing for multiple batches
        if len(batches) > 1 and self._concurrent_batches > 1:
            return self._embed_concurrent(client, batches)
        else:
            return self._embed_sequential(client, batches)

    def _embed_sequential(
        self, client: Any, batches: list[tuple[int, list[str]]]
    ) -> np.ndarray:
        """Process batches sequentially with adaptive rate limiting.

        Args:
            client: Voyage client instance.
            batches: List of (start_index, texts) tuples.

        Returns:
            NumPy array of all embeddings.
        """
        # Pre-calculate total size for efficient allocation
        total_items = sum(len(batch) for _, batch in batches)
        all_embeddings = np.zeros((total_items, self.dimension), dtype=np.float32)
        idx = 0

        for i, (_, batch) in enumerate(batches):
            embeddings, response_time = self._embed_batch_with_timing(client, batch)

            # Direct NumPy array assignment (avoid list intermediate)
            batch_arr = np.asarray(embeddings, dtype=np.float32)
            n = len(batch_arr)
            all_embeddings[idx : idx + n] = batch_arr
            idx += n

            # Adaptive rate limiting
            if i < len(batches) - 1:  # Not the last batch
                delay = self._calculate_delay(response_time)
                if delay > 0:
                    time.sleep(delay)

        return all_embeddings

    def _embed_concurrent(
        self, client: Any, batches: list[tuple[int, list[str]]]
    ) -> np.ndarray:
        """Process batches concurrently for faster throughput.

        Args:
            client: Voyage client instance.
            batches: List of (start_index, texts) tuples.

        Returns:
            NumPy array of all embeddings in correct order.
        """
        # Pre-allocate result array
        total_items = sum(len(batch) for _, batch in batches)
        all_embeddings = np.zeros((total_items, self.dimension), dtype=np.float32)

        # Track batch sizes for proper indexing
        batch_info: dict[
            int, tuple[int, int]
        ] = {}  # start_idx -> (output_offset, size)
        offset = 0
        for start_idx, batch in batches:
            batch_info[start_idx] = (offset, len(batch))
            offset += len(batch)

        results: dict[int, np.ndarray] = {}

        with ThreadPoolExecutor(max_workers=self._concurrent_batches) as executor:
            futures = {
                executor.submit(self._embed_batch_with_timing, client, batch): idx
                for idx, batch in batches
            }

            for future in as_completed(futures):
                start_idx = futures[future]
                embeddings, response_time = future.result()
                # Convert to NumPy immediately
                results[start_idx] = np.asarray(embeddings, dtype=np.float32)
                self._last_response_time = response_time

        # Reconstruct in correct order using direct array assignment
        for start_idx, (output_offset, size) in batch_info.items():
            all_embeddings[output_offset : output_offset + size] = results[start_idx]

        return all_embeddings

    def _calculate_delay(self, response_time: float) -> float:
        """Calculate delay based on API response time.

        Args:
            response_time: Time taken for last API call.

        Returns:
            Delay in seconds before next call.
        """
        if not self._adaptive_rate_limit:
            return VOYAGE_RATE_LIMIT_MIN_DELAY

        if response_time < VOYAGE_RATE_LIMIT_FAST_THRESHOLD:
            # Fast response = API is not under pressure, minimal delay
            return VOYAGE_RATE_LIMIT_MIN_DELAY
        elif response_time < VOYAGE_RATE_LIMIT_MODERATE_THRESHOLD:
            # Moderate response = slight back-pressure
            return VOYAGE_RATE_LIMIT_MODERATE_DELAY
        else:
            # Slow response = already rate limited, no additional delay
            return 0.0

    def _embed_batch_with_timing(
        self, client: Any, texts: list[str]
    ) -> tuple[list[list[float]], float]:
        """Embed batch and return response time.

        Args:
            client: Voyage client instance.
            texts: Batch of texts to embed.

        Returns:
            Tuple of (embeddings, response_time_seconds).
        """
        start = time.perf_counter()
        embeddings = self._embed_batch_with_retry(client, texts)
        elapsed = time.perf_counter() - start
        self._last_response_time = elapsed
        return embeddings, elapsed

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
                    time.sleep(VOYAGE_RETRY_BASE**attempt)

        raise last_error or RuntimeError("Unknown error in Voyage API")


__all__ = ["VoyageBackend"]
