"""Local sentence-transformers embedding backend.

Uses sentence-transformers for offline embedding generation.
Provides fallback when Voyage API is unavailable.
"""

import logging
from typing import Any

import numpy as np

from .base import EmbeddingBackend

logger = logging.getLogger(__name__)

# Default model for general-purpose embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Model dimensions for common models
MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "all-distilroberta-v1": 768,
}


class LocalBackend(EmbeddingBackend):
    """Local sentence-transformers embedding backend.

    Uses sentence-transformers library for offline embedding generation.
    Supports GPU acceleration when CUDA is available.

    Example:
        >>> backend = LocalBackend()
        >>> vectors = backend.embed(["login form", "dashboard"])
        >>> print(vectors.shape)
        (2, 384)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize: bool = True,
    ):
        """Initialize local backend.

        Args:
            model_name: sentence-transformers model name.
            device: Device for computation ('cuda', 'cpu', or None for auto).
            normalize: Whether to normalize embeddings for cosine similarity.

        Raises:
            ImportError: If sentence-transformers not installed.
        """
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension for configured model."""
        if self._dimension is not None:
            return self._dimension

        # Check known dimensions first
        if self._model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self._model_name]

        # Load model to get dimension
        model = self._get_model()
        self._dimension = model.get_sentence_embedding_dimension()
        return self._dimension

    @property
    def name(self) -> str:
        """Get backend identifier."""
        return f"local:{self._model_name}"

    def _get_model(self) -> Any:
        """Lazily initialize sentence-transformers model.

        Returns:
            Initialized SentenceTransformer instance.

        Raises:
            ImportError: If sentence-transformers not installed.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name, device=self._device)
                logger.info(
                    f"Loaded model '{self._model_name}' on device "
                    f"'{self._model.device}'"
                )
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers required for LocalBackend. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            NumPy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for single query.

        Args:
            query: Query text to embed.

        Returns:
            NumPy array of shape (dimension,).
        """
        result = self.embed([query])
        return result[0]


__all__ = ["LocalBackend"]
