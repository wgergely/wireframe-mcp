"""Local sentence-transformers embedding backend.

Uses sentence-transformers for offline embedding generation.
Provides fallback when Voyage API is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..models import get_model_manager
from .base import EmbeddingBackend
from .model_spec import DEFAULT_LOCAL_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


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
        model_name: str = DEFAULT_LOCAL_MODEL.spec.name,
        device: str | None = None,
        normalize: bool = True,
        models_dir: Path | str | None = None,
        auto_download: bool = True,
    ):
        """Initialize local backend.

        Args:
            model_name: sentence-transformers model name.
            device: Device for computation ('cuda', 'cpu', or None for auto).
            normalize: Whether to normalize embeddings for cosine similarity.
            models_dir: Override path for model storage. Uses centralized
                .corpus/models directory if not provided.
            auto_download: Automatically download model if not present.

        Raises:
            ImportError: If sentence-transformers not installed.
        """
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._auto_download = auto_download
        self._model_manager = get_model_manager(models_dir)
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension for configured model."""
        if self._dimension is not None:
            return self._dimension

        # Check known dimensions from registry first
        dim = self._model_manager.get_dimension(self._model_name)
        if dim is not None:
            return dim

        # Load model to get dimension
        model = self._get_model()
        self._dimension = model.get_sentence_embedding_dimension()
        return self._dimension

    @property
    def name(self) -> str:
        """Get backend identifier."""
        return f"local:{self._model_name}"

    def _get_model(self) -> SentenceTransformer:
        """Lazily initialize sentence-transformers model.

        Uses the centralized ModelManager for model storage and loading.
        Models are stored in .corpus/models by default.

        Returns:
            Initialized SentenceTransformer instance.

        Raises:
            ImportError: If sentence-transformers not installed.
            FileNotFoundError: If model not found and auto_download=False.
        """
        if self._model is None:
            self._model = self._model_manager.load(
                self._model_name,
                device=self._device,
                auto_download=self._auto_download,
            )
            logger.info(
                f"Loaded model '{self._model_name}' on device '{self._model.device}'"
            )
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
