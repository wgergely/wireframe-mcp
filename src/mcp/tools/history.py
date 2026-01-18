"""History management tools for MCP server.

These tools provide access to generation history, artifacts, and variations.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_history(
    session_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    query: str | None = None,
) -> dict[str, Any]:
    """Retrieve generation history.

    Lists past generations with optional filtering by session or
    semantic search by query.

    Args:
        session_id: Filter by session ID.
        limit: Maximum results (1-100). Default: 20
        offset: Number of results to skip. Default: 0
        query: Semantic search query (optional).

    Returns:
        Dictionary containing:
        - artifacts: List of artifact summaries
        - total_count: Total matching artifacts
        - has_more: Whether more results exist

    Example:
        >>> result = get_history(limit=10)
        >>> for artifact in result["artifacts"]:
        ...     print(f"{artifact['id']}: {artifact['query']}")
    """
    from src.history import get_history_manager

    # Validate parameters
    limit = max(1, min(100, limit))
    offset = max(0, offset)

    manager = get_history_manager()

    if query:
        # Semantic search
        results = manager.search_artifacts(
            query=query,
            session_id=session_id,
            limit=limit,
        )
        artifacts = [
            _artifact_to_summary(artifact, score) for artifact, score in results
        ]
        total_count = len(artifacts)
    else:
        # List artifacts
        artifacts_list = manager.list_artifacts(
            session_id=session_id,
            limit=limit + 1,  # +1 to check if more exist
            offset=offset,
        )
        has_more = len(artifacts_list) > limit
        artifacts_list = artifacts_list[:limit]
        artifacts = [_artifact_to_summary(a) for a in artifacts_list]
        total_count = offset + len(artifacts) + (1 if has_more else 0)

    return {
        "artifacts": artifacts,
        "total_count": total_count,
        "has_more": len(artifacts) == limit,
    }


def get_artifact(
    artifact_id: str,
    include_lineage: bool = False,
) -> dict[str, Any]:
    """Retrieve a specific artifact by ID.

    Args:
        artifact_id: The artifact UUID.
        include_lineage: Include parent/child relationships. Default: False

    Returns:
        Dictionary containing:
        - artifact: Full artifact data with layout, draft, stats
        - lineage: Parent/child IDs (if include_lineage=True)

    Raises:
        ValueError: If artifact not found.

    Example:
        >>> result = get_artifact("abc-123")
        >>> print(result["artifact"]["draft"])
    """
    from src.history import get_history_manager

    manager = get_history_manager()
    artifact = manager.get_artifact(artifact_id)

    if not artifact:
        raise ValueError(f"Artifact not found: {artifact_id}")

    result: dict[str, Any] = {
        "artifact": _artifact_to_full(artifact),
    }

    if include_lineage:
        lineage = manager.get_lineage(artifact_id)
        result["lineage"] = {
            "ancestors": [a.id for a in lineage["ancestors"]],
            "descendants": [a.id for a in lineage["descendants"]],
        }

    return result


def get_sessions(
    limit: int = 20,
    offset: int = 0,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """List available sessions.

    Args:
        limit: Maximum results (1-50). Default: 20
        offset: Number of results to skip. Default: 0
        tags: Filter by tags (any match).

    Returns:
        Dictionary containing:
        - sessions: List of session summaries
        - total_count: Total sessions

    Example:
        >>> result = get_sessions(limit=5)
        >>> for session in result["sessions"]:
        ...     print(f"{session['name']}: {session['artifact_count']} artifacts")
    """
    from src.history import get_history_manager

    limit = max(1, min(50, limit))
    offset = max(0, offset)

    manager = get_history_manager()
    sessions = manager.list_sessions(limit=limit, offset=offset, tags=tags)

    return {
        "sessions": [_session_to_summary(s) for s in sessions],
        "total_count": len(sessions),
    }


def get_variation_set(
    set_id: str,
) -> dict[str, Any]:
    """Retrieve a variation set with all its artifacts.

    Args:
        set_id: The variation set UUID.

    Returns:
        Dictionary containing:
        - variation_set: Set metadata
        - artifacts: List of artifact summaries
        - comparison: Comparison metrics

    Raises:
        ValueError: If variation set not found.
    """
    from src.history import compare_artifacts, get_history_manager

    manager = get_history_manager()
    variation_set = manager.get_variation_set(set_id)

    if not variation_set:
        raise ValueError(f"Variation set not found: {set_id}")

    artifacts = manager.get_variation_artifacts(set_id)
    comparison = compare_artifacts(artifacts)

    return {
        "variation_set": {
            "id": variation_set.id,
            "session_id": variation_set.session_id,
            "query": variation_set.request.query,
            "count": variation_set.request.count,
            "diversity_score": variation_set.diversity_score,
            "rankings": variation_set.rankings,
            "created_at": variation_set.created_at.isoformat(),
        },
        "artifacts": [_artifact_to_summary(a) for a in artifacts],
        "comparison": comparison,
    }


def delete_artifact(
    artifact_id: str,
) -> dict[str, Any]:
    """Delete an artifact from history.

    Args:
        artifact_id: The artifact UUID to delete.

    Returns:
        Dictionary with:
        - deleted: True if deleted
        - artifact_id: ID of deleted artifact

    Raises:
        ValueError: If artifact not found.
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    if not manager.get_artifact(artifact_id):
        raise ValueError(f"Artifact not found: {artifact_id}")

    deleted = manager.delete_artifact(artifact_id)

    return {
        "deleted": deleted,
        "artifact_id": artifact_id,
    }


def cleanup_history(
    delete_orphans: bool = True,
    delete_stale: bool = True,
    enforce_limits: bool = True,
) -> dict[str, Any]:
    """Clean up history storage.

    Removes orphaned artifacts, stale data, and enforces size limits.

    Args:
        delete_orphans: Delete artifacts without sessions. Default: True
        delete_stale: Delete artifacts exceeding max age. Default: True
        enforce_limits: Enforce size and count limits. Default: True

    Returns:
        Dictionary with cleanup results:
        - artifacts_deleted: Number deleted
        - sessions_deleted: Number deleted
        - bytes_freed: Bytes freed
        - mb_freed: MB freed
    """
    from src.history import get_history_manager

    manager = get_history_manager()
    result = manager.cleanup(
        delete_orphans=delete_orphans,
        delete_stale=delete_stale,
        enforce_size_limit=enforce_limits,
        enforce_count_limit=enforce_limits,
    )

    return {
        "artifacts_deleted": result.artifacts_deleted,
        "sessions_deleted": result.sessions_deleted,
        "orphans_cleaned": result.orphans_cleaned,
        "previews_deleted": result.previews_deleted,
        "bytes_freed": result.bytes_freed,
        "mb_freed": result.mb_freed,
        "errors": result.errors,
    }


def get_storage_stats() -> dict[str, Any]:
    """Get history storage statistics.

    Returns:
        Dictionary with storage stats:
        - total_size_mb: Total storage size
        - artifact_count: Number of artifacts
        - session_count: Number of sessions
        - oldest_artifact_days: Age of oldest artifact
        - orphan_count: Number of orphaned artifacts
    """
    from src.history import get_history_manager

    manager = get_history_manager()
    stats = manager.get_stats()

    return {
        "total_size_mb": stats.total_size_mb,
        "total_size_bytes": stats.total_size_bytes,
        "artifact_count": stats.artifact_count,
        "session_count": stats.session_count,
        "variation_set_count": stats.variation_set_count,
        "preview_cache_size_mb": stats.preview_cache_size_mb,
        "oldest_artifact_days": stats.oldest_artifact_age_days,
        "orphan_count": stats.orphan_count,
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _artifact_to_summary(
    artifact: Any,
    score: float | None = None,
) -> dict[str, Any]:
    """Convert artifact to summary dict."""
    summary = {
        "id": artifact.id,
        "query": artifact.query,
        "draft_preview": artifact.draft[:200] + "..."
        if len(artifact.draft) > 200
        else artifact.draft,
        "model": artifact.model,
        "temperature": artifact.temperature,
        "created_at": artifact.created_at.isoformat(),
        "tags": artifact.tags,
        "variation_group": artifact.variation_group,
        "variation_index": artifact.variation_index,
    }
    if score is not None:
        summary["relevance_score"] = score
    return summary


def _artifact_to_full(artifact: Any) -> dict[str, Any]:
    """Convert artifact to full dict."""
    return {
        "id": artifact.id,
        "session_id": artifact.session_id,
        "parent_id": artifact.parent_id,
        "query": artifact.query,
        "layout": artifact.layout,
        "draft": artifact.draft,
        "model": artifact.model,
        "temperature": artifact.temperature,
        "provider": artifact.provider,
        "stats": {
            "attempts": artifact.stats.attempts,
            "validation_retries": artifact.stats.validation_retries,
            "total_tokens": artifact.stats.total_tokens,
            "final_model": artifact.stats.final_model,
        },
        "rag_example_ids": artifact.rag_example_ids,
        "rag_scores": artifact.rag_scores,
        "variation_group": artifact.variation_group,
        "variation_index": artifact.variation_index,
        "tags": artifact.tags,
        "status": artifact.status.value,
        "size_bytes": artifact.size_bytes,
        "created_at": artifact.created_at.isoformat(),
        "accessed_at": artifact.accessed_at.isoformat(),
    }


def _session_to_summary(session: Any) -> dict[str, Any]:
    """Convert session to summary dict."""
    return {
        "id": session.id,
        "name": session.name,
        "description": session.description,
        "artifact_count": session.artifact_count,
        "total_tokens": session.total_tokens,
        "size_mb": session.size_bytes / (1024 * 1024),
        "tags": session.tags,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


__all__ = [
    "get_history",
    "get_artifact",
    "get_sessions",
    "get_variation_set",
    "delete_artifact",
    "cleanup_history",
    "get_storage_stats",
]
