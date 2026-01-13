"""Vector database module for UI layout similarity search.

This module provides:
- Schema-aware LayoutNode serialization for embeddings
- Pluggable embedding backends (Voyage AI, local sentence-transformers)
- FAISS-based vector index with GPU acceleration
- VectorStore API for indexing and searching corpus data

Example usage:
    >>> from src.vector import VectorStore
    >>> store = VectorStore(backend="voyage")
    >>> store.index_corpus(provider)
    >>> results = store.search("login form with social auth")
"""

from .serializer import (
    LayoutSerializer,
    SerializationConfig,
    SerializationFormat,
    SerializedLayout,
    serialize_layout,
)

__all__ = [
    # Serialization
    "LayoutSerializer",
    "SerializationConfig",
    "SerializationFormat",
    "SerializedLayout",
    "serialize_layout",
]
