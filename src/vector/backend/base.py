"""Abstract base class for embedding backends."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    """Abstract interface for embedding generation.

    Embedding backends convert text into dense vector representations
    suitable for similarity search. Implementations may use external
    APIs (Voyage) or local models (sentence-transformers).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            NumPy array of shape (len(texts), dimension) with embeddings.
            Vectors should be normalized for cosine similarity.
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query.

        Some backends use different models for queries vs documents.
        Default implementation calls embed() with single-item list.

        Args:
            query: Query text to embed.

        Returns:
            NumPy array of shape (dimension,) with query embedding.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding vector dimension.

        Returns:
            Integer dimension of output vectors.
        """

    @property
    def name(self) -> str:
        """Get backend name for logging/stats.

        Returns:
            String identifier for this backend.
        """
        return self.__class__.__name__


__all__ = ["EmbeddingBackend"]
