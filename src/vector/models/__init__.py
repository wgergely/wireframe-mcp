"""Centralized embedding model management.

Provides unified storage and retrieval for local embedding models.
Models are stored in {repo_root}/.corpus/models by default.

Example:
    >>> from src.vector.models import get_model_manager
    >>>
    >>> manager = get_model_manager()
    >>> manager.download("all-MiniLM-L6-v2")
    >>> model = manager.load("all-MiniLM-L6-v2")
"""

from .lib import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    ModelManager,
    get_model_manager,
)

__all__ = [
    "ModelManager",
    "get_model_manager",
    "DEFAULT_MODEL",
    "MODEL_REGISTRY",
]
