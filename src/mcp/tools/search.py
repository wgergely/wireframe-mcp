"""Search layouts tool for MCP server.

This tool searches the vector database for similar layouts.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def search_layouts(
    query: str,
    k: int = 5,
    source_filter: str | None = None,
) -> dict[str, Any]:
    """Search for similar layouts in the vector database.

    Finds layouts in the corpus that are semantically similar to the
    query. Useful for finding inspiration or understanding existing
    patterns before generating new layouts.

    Args:
        query: Natural language description of desired layout.
            Examples:
            - "settings page with toggle switches"
            - "dashboard with charts and metrics"
            - "user profile with avatar and bio"
        k: Number of results to return (1-20). Default: 5
        source_filter: Optional filter by corpus source.
            Options: "rico", "enrico", "websight", "showui", "egfe"

    Returns:
        Dictionary containing:
        - results: List of similar layouts with scores and metadata
        - total_in_index: Total number of items in the index
        - query: The original query

    Raises:
        ToolError: If vector index is not available.

    Example:
        >>> result = search_layouts("login form with social buttons", k=3)
        >>> for item in result["results"]:
        ...     print(f"Score: {item['score']:.2f} - {item['text'][:50]}...")
    """
    from src.config import get_index_dir
    from src.vector import VectorStore

    # Clamp k to reasonable range
    k = max(1, min(20, k))

    # Load vector store
    index_dir = get_index_dir()
    if not (index_dir / "index.faiss").exists():
        raise RuntimeError(
            f"Vector index not found at {index_dir}. "
            "Build it with: python . dev index build"
        )

    store = VectorStore()
    store.load(index_dir)

    logger.info(f"Searching for: {query} (k={k})")

    # Search
    results = store.search(query, k=k, source_filter=source_filter)

    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append(
            {
                "score": round(result.score, 4),
                "text": result.text,
                "metadata": {
                    "id": result.item_id,
                    "source": result.metadata.get("source", "unknown"),
                    **{k: v for k, v in result.metadata.items() if k != "source"},
                },
            }
        )

    return {
        "results": formatted_results,
        "total_in_index": len(store),
        "query": query,
    }


__all__ = ["search_layouts"]
