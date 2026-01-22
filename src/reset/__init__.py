"""Environment reset utilities for wireframe-mcp.

Provides functions to clear development artifacts:
- RAG vector indices
- Downloaded embedding models
- Docker containers, volumes, and images
- Temporary files and caches

Example:
    >>> from src.reset import reset_environment
    >>> reset_environment(index=True, docker=True)
"""

from .lib import (
    clear_indices,
    clear_models,
    clear_docker,
    clear_temp,
    reset_environment,
)

__all__ = [
    "clear_indices",
    "clear_models",
    "clear_docker",
    "clear_temp",
    "reset_environment",
]
