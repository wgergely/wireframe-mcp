"""Centralized embedding model management.

Provides unified storage and retrieval for local embedding models.
Models are stored in {repo_root}/.corpus/models by default.

Example:
    >>> from src.vector.models import get_model_manager, DEFAULT_LOCAL_MODEL
    >>>
    >>> manager = get_model_manager()
    >>> manager.download(DEFAULT_LOCAL_MODEL.spec.name)
    >>> model = manager.load(DEFAULT_LOCAL_MODEL.spec.name)
"""

from .lib import (
    DEFAULT_LOCAL_MODEL,
    EmbeddingModel,
    ModelManager,
    ModelSpec,
    get_model_manager,
)

__all__ = [
    "ModelManager",
    "get_model_manager",
    "DEFAULT_LOCAL_MODEL",
    "EmbeddingModel",
    "ModelSpec",
]
