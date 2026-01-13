"""Embedding backend implementations.

Provides pluggable embedding backends for vector generation:
- VoyageBackend: Voyage AI API (recommended, large context)
- LocalBackend: sentence-transformers (offline fallback)

Use `create_backend()` for easy provider switching:
    >>> from src.vector.backend import create_backend, EmbeddingModel
    >>> backend = create_backend(EmbeddingModel.VOYAGE_3)
"""

from .base import EmbeddingBackend
from .factory import create_backend
from .local import LocalBackend
from .model_spec import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_VOYAGE_MODEL,
    EmbeddingModel,
    ModelCapability,
    ModelSpec,
    ProviderType,
    get_model_spec,
)
from .voyage import VoyageBackend

__all__ = [
    # Base class
    "EmbeddingBackend",
    # Implementations
    "VoyageBackend",
    "LocalBackend",
    # Model specification system
    "ModelSpec",
    "ModelCapability",
    "ProviderType",
    "EmbeddingModel",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_VOYAGE_MODEL",
    "get_model_spec",
    # Factory
    "create_backend",
]

