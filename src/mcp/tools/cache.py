"""Cached resources for MCP tools.

Provides cached access to expensive resources like VectorStore
to avoid repeated disk I/O and initialization on each tool call.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store():
    """Get cached VectorStore instance.

    Returns:
        VectorStore if index exists and loads successfully, None otherwise.

    Note:
        The cache persists for the lifetime of the server process.
        If the index is rebuilt, the server should be restarted.
    """
    from src.config import get_index_dir
    from src.vector import VectorStore

    index_dir = get_index_dir()
    index_path = index_dir / "index.faiss"

    if not index_path.exists():
        logger.debug(f"No vector index found at {index_dir}")
        return None

    try:
        store = VectorStore()
        store.load(index_dir)
        logger.info(f"Loaded vector index with {len(store)} items")
        return store
    except Exception as e:
        logger.warning(f"Failed to load vector index: {e}")
        return None


def clear_vector_store_cache() -> None:
    """Clear the VectorStore cache.

    Call this if the index has been rebuilt and needs to be reloaded.
    """
    get_vector_store.cache_clear()
    logger.info("Vector store cache cleared")


__all__ = ["get_vector_store", "clear_vector_store_cache"]
