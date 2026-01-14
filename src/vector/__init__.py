"""Vector database module for UI layout similarity search.

This module provides:
- Schema-aware LayoutNode serialization for embeddings
- Pluggable embedding backends (Voyage AI, local sentence-transformers)
- FAISS-based vector index with GPU acceleration (prioritized)
- VectorStore API for indexing and searching corpus data

GPU Support:
    FAISS-GPU is automatically used when available for optimal performance.
    Falls back to CPU if CUDA is not detected. Override with:
    - VECTOR_USE_GPU=true  (force GPU)
    - VECTOR_USE_GPU=false (force CPU)

Example usage:
    >>> from src.vector import VectorStore, BackendType
    >>> store = VectorStore(backend=BackendType.VOYAGE)
    >>> store.index_corpus(provider)
    >>> results = store.search("login form with social auth")
"""

from .backend import EmbeddingBackend, LocalBackend, VoyageBackend
from .index import FAISSIndex, SearchResult
from .lib import IndexStats, VectorSearchResult, VectorStore
from .serializer import (
    LayoutSerializer,
    SerializationConfig,
    SerializationFormat,
    SerializedLayout,
    serialize_layout,
)
from .types import (
    BackendType,
    EnvVar,
    ItemMetadata,
    MetadataKey,
    OrientationCode,
)

__all__ = [
    # Types and enums
    "BackendType",
    "EnvVar",
    "ItemMetadata",
    "MetadataKey",
    "OrientationCode",
    # Serialization
    "LayoutSerializer",
    "SerializationConfig",
    "SerializationFormat",
    "SerializedLayout",
    "serialize_layout",
    # Embedding backends
    "EmbeddingBackend",
    "VoyageBackend",
    "LocalBackend",
    # FAISS index
    "FAISSIndex",
    "SearchResult",
    # VectorStore
    "VectorStore",
    "IndexStats",
    "VectorSearchResult",
]
