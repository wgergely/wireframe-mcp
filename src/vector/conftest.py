"""Vector module test fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# FAISS Availability Check
# =============================================================================

try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")


# =============================================================================
# Mock Backend
# =============================================================================


class MockEmbeddingBackend:
    """Mock embedding backend for deterministic testing.

    Produces reproducible embeddings based on text hash for testing
    without requiring actual embedding models.
    """

    def __init__(self, dimension: int = 64):
        """Initialize mock backend.

        Args:
            dimension: Embedding dimension to use.
        """
        self._dimension = dimension
        self._embed_count = 0

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def name(self) -> str:
        """Return backend name."""
        return "mock:test"

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate deterministic embeddings from text.

        Args:
            texts: List of texts to embed.

        Returns:
            Numpy array of shape (len(texts), dimension).
        """
        self._embed_count += len(texts)
        embeddings = np.zeros((len(texts), self._dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            # Use text hash as seed for reproducibility
            seed = hash(text) % (2**32)
            rng = np.random.default_rng(seed)
            embeddings[i] = rng.standard_normal(self._dimension).astype(np.float32)
            # Normalize to unit length
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Embed a single query.

        Args:
            query: Query text to embed.

        Returns:
            1D numpy array of embeddings.
        """
        return self.embed([query])[0]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_backend() -> MockEmbeddingBackend:
    """Create a mock embedding backend for testing.

    Returns:
        MockEmbeddingBackend instance with 64 dimensions.
    """
    return MockEmbeddingBackend(dimension=64)


@pytest.fixture
def sample_vectors() -> tuple[NDArray[np.float32], list[str]]:
    """Create sample vectors and IDs for FAISS testing.

    Returns:
        Tuple of (vectors array, list of IDs).
    """
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((10, 64)).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    ids = [f"item_{i}" for i in range(10)]
    return vectors, ids


@pytest.fixture
def large_vectors() -> tuple[NDArray[np.float32], list[str]]:
    """Create larger vector set for batch testing.

    Returns:
        Tuple of (vectors array, list of IDs) with 100 items.
    """
    rng = np.random.default_rng(123)
    vectors = rng.standard_normal((100, 64)).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    ids = [f"batch_item_{i}" for i in range(100)]
    return vectors, ids


@pytest.fixture
def sample_layout_texts() -> list[str]:
    """Sample layout serialization texts for embedding tests.

    Returns:
        List of serialized layout strings.
    """
    return [
        "container[vertical] > toolbar + card[content] + navbar",
        "container[horizontal] > drawer[sidebar] + container[main]",
        "card > text[title] + input[email] + input[password] + button[submit]",
        "container > toolbar[header] > button[menu] + text[title]",
        "list > list_item[item_1] + list_item[item_2] + list_item[item_3]",
    ]
