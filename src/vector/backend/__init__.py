"""Embedding backend implementations.

Provides pluggable embedding backends for vector generation:
- VoyageBackend: Voyage AI API (recommended, large context)
- LocalBackend: sentence-transformers (offline fallback)
"""

from .base import EmbeddingBackend
from .local import LocalBackend
from .voyage import VoyageBackend

__all__ = [
    "EmbeddingBackend",
    "VoyageBackend",
    "LocalBackend",
]
