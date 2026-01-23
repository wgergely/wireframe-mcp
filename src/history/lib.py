"""History Manager for wireframe-mcp.

Provides artifact persistence, session management, variation generation,
and lineage tracking for generated layouts.
"""

import logging
from pathlib import Path
from typing import Any

from .models import (
    CleanupResult,
    GenerationArtifact,
    GenerationStats,
    Interaction,
    Session,
    StorageConfig,
    StorageStats,
    VariationRequest,
    VariationSet,
)
from .storage import SQLiteStorage
from .storage.protocol import HistoryStorage

logger = logging.getLogger(__name__)

# Default storage path
DEFAULT_DB_PATH = Path("data/history/history.db")


class HistoryManager:
    """Manager for generation history and artifacts.

    Provides a high-level interface for:
    - Session management (grouping related operations)
    - Artifact storage and retrieval
    - Variation generation
    - Lineage tracking
    - Storage size management with automatic cleanup

    Example:
        >>> manager = HistoryManager()
        >>> session = manager.get_or_create_session("my-project")
        >>> artifact = manager.store_artifact(
        ...     session_id=session.id,
        ...     query="login form",
        ...     layout={"type": "container", ...},
        ...     draft="Login [container]\\n└── Form [card]",
        ... )
        >>> print(f"Stored artifact: {artifact.id}")

    Args:
        storage: Storage backend to use. If None, creates SQLiteStorage.
        config: Storage configuration. If None, uses defaults.
        db_path: Path to database file (only used if storage is None).
        auto_cleanup: Whether to run cleanup on initialization.
    """

    def __init__(
        self,
        storage: HistoryStorage | None = None,
        config: StorageConfig | None = None,
        db_path: Path | str | None = None,
        auto_cleanup: bool = True,
    ):
        self._config = config or StorageConfig()

        if storage:
            self._storage = storage
        else:
            db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
            self._storage = SQLiteStorage(db_path, self._config)

        self._storage.initialize()
        self._current_session: Session | None = None

        if auto_cleanup:
            self._run_startup_cleanup()

    def _run_startup_cleanup(self) -> None:
        """Run cleanup on startup to maintain storage limits."""
        try:
            result = self._storage.cleanup(
                delete_orphans=True,
                delete_stale=True,
                enforce_size_limit=True,
                enforce_count_limit=True,
            )
            if result.artifacts_deleted > 0:
                logger.info(
                    f"Startup cleanup: removed {result.artifacts_deleted} artifacts, "
                    f"freed {result.mb_freed:.2f} MB"
                )
        except Exception as e:
            logger.warning(f"Startup cleanup failed: {e}")

    @property
    def config(self) -> StorageConfig:
        """Get storage configuration."""
        return self._config

    def close(self) -> None:
        """Close storage connections."""
        self._storage.close()

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(
        self,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> Session:
        """Create a new session for grouping related operations.

        Args:
            name: Optional user-friendly name.
            description: Optional description.
            tags: Optional tags for organization.

        Returns:
            Created session.
        """
        session = Session.create(name=name, description=description, tags=tags)
        return self._storage.create_session(session)

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session if found, None otherwise.
        """
        return self._storage.get_session(session_id)

    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        tags: list[str] | None = None,
    ) -> list[Session]:
        """List sessions with optional filtering.

        Args:
            limit: Maximum sessions to return.
            offset: Number to skip.
            tags: Filter by tags.

        Returns:
            List of sessions ordered by updated_at descending.
        """
        return self._storage.list_sessions(limit=limit, offset=offset, tags=tags)

    def get_or_create_session(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Session:
        """Get the current session or create a new one.

        If no current session exists, creates a new one with the given
        name and description.

        Args:
            name: Name for new session if created.
            description: Description for new session if created.

        Returns:
            Current or newly created session.
        """
        if self._current_session:
            # Refresh from storage
            refreshed = self._storage.get_session(self._current_session.id)
            if refreshed:
                self._current_session = refreshed
                return self._current_session

        # Create new session
        self._current_session = self.create_session(name=name, description=description)
        return self._current_session

    def set_current_session(self, session_id: str) -> Session | None:
        """Set the current working session.

        Args:
            session_id: Session to make current.

        Returns:
            Session if found, None otherwise.
        """
        session = self._storage.get_session(session_id)
        if session:
            self._current_session = session
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts.

        Args:
            session_id: Session to delete.

        Returns:
            True if deleted, False if not found.
        """
        if self._current_session and self._current_session.id == session_id:
            self._current_session = None
        return self._storage.delete_session(session_id)

    # =========================================================================
    # Artifact Storage
    # =========================================================================

    def store_artifact(
        self,
        query: str,
        layout: dict[str, Any],
        draft: str,
        session_id: str | None = None,
        parent_id: str | None = None,
        model: str = "",
        temperature: float = 0.7,
        provider: str = "d2",
        stats: GenerationStats | None = None,
        rag_example_ids: list[str] | None = None,
        rag_scores: dict[str, float] | None = None,
        variation_group: str | None = None,
        variation_index: int | None = None,
        tags: list[str] | None = None,
    ) -> GenerationArtifact:
        """Store a generated layout as an artifact.

        Args:
            query: Original natural language query.
            layout: Generated layout as JSON dict.
            draft: Text tree representation.
            session_id: Session to attach to (uses current if None).
            parent_id: Parent artifact for refinements.
            model: LLM model used.
            temperature: Generation temperature.
            provider: DSL provider hint.
            stats: Generation statistics.
            rag_example_ids: IDs of RAG examples used.
            rag_scores: Relevance scores for RAG examples.
            variation_group: Group ID for variations.
            variation_index: Index within variation set.
            tags: User-defined tags.

        Returns:
            Stored artifact with generated ID.
        """
        # Use current session if not specified
        if session_id is None:
            session = self.get_or_create_session()
            session_id = session.id

        artifact = GenerationArtifact.create(
            session_id=session_id,
            query=query,
            layout=layout,
            draft=draft,
            parent_id=parent_id,
            model=model,
            temperature=temperature,
            provider=provider,
            stats=stats or GenerationStats(),
            rag_example_ids=rag_example_ids or [],
            rag_scores=rag_scores or {},
            variation_group=variation_group,
            variation_index=variation_index,
            tags=tags or [],
        )

        stored = self._storage.store_artifact(artifact)
        logger.debug(f"Stored artifact {stored.id} ({stored.size_bytes} bytes)")
        return stored

    def get_artifact(self, artifact_id: str) -> GenerationArtifact | None:
        """Get an artifact by ID.

        Also updates the artifact's accessed_at timestamp.

        Args:
            artifact_id: Artifact identifier.

        Returns:
            Artifact if found, None otherwise.
        """
        artifact = self._storage.get_artifact(artifact_id)
        if artifact:
            self._storage.touch_artifact(artifact_id)
        return artifact

    def list_artifacts(
        self,
        session_id: str | None = None,
        parent_id: str | None = None,
        variation_group: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GenerationArtifact]:
        """List artifacts with filtering options.

        Args:
            session_id: Filter by session.
            parent_id: Filter by parent artifact.
            variation_group: Filter by variation group.
            tags: Filter by tags.
            limit: Maximum artifacts to return.
            offset: Number to skip.

        Returns:
            List of artifacts ordered by created_at descending.
        """
        return self._storage.list_artifacts(
            session_id=session_id,
            parent_id=parent_id,
            variation_group=variation_group,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    def search_artifacts(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[GenerationArtifact, float]]:
        """Search artifacts by query similarity.

        Args:
            query: Search query.
            session_id: Limit to session.
            limit: Maximum results.

        Returns:
            List of (artifact, score) tuples.
        """
        return self._storage.search_artifacts_by_query(
            query=query,
            session_id=session_id,
            limit=limit,
        )

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: Artifact to delete.

        Returns:
            True if deleted, False if not found.
        """
        return self._storage.delete_artifact(artifact_id)

    # =========================================================================
    # Lineage Tracking
    # =========================================================================

    def get_children(self, artifact_id: str) -> list[GenerationArtifact]:
        """Get all artifacts derived from this one.

        Args:
            artifact_id: Parent artifact ID.

        Returns:
            List of child artifacts.
        """
        return self._storage.get_artifact_children(artifact_id)

    def get_lineage(self, artifact_id: str) -> dict[str, list[GenerationArtifact]]:
        """Get full lineage (ancestors and descendants).

        Args:
            artifact_id: Artifact to trace.

        Returns:
            Dict with 'ancestors' and 'descendants' lists.
        """
        return self._storage.get_artifact_lineage(artifact_id)

    def get_rag_influence(
        self,
        artifact_id: str,
    ) -> list[tuple[str, float]]:
        """Get RAG examples that influenced a generation.

        Args:
            artifact_id: Artifact to check.

        Returns:
            List of (example_id, score) tuples.
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return []

        return [
            (example_id, artifact.rag_scores.get(example_id, 0.0))
            for example_id in artifact.rag_example_ids
        ]

    # =========================================================================
    # Variation Sets
    # =========================================================================

    def store_variation_set(
        self,
        request: VariationRequest,
        artifact_ids: list[str],
        diversity_score: float = 0.0,
        rankings: dict[str, list[str]] | None = None,
        session_id: str | None = None,
    ) -> VariationSet:
        """Store a variation set.

        Args:
            request: Original variation request.
            artifact_ids: IDs of generated artifacts.
            diversity_score: Calculated diversity metric.
            rankings: Criterion -> ordered artifact IDs.
            session_id: Session to attach to.

        Returns:
            Stored variation set.
        """
        if session_id is None:
            session = self.get_or_create_session()
            session_id = session.id

        variation_set = VariationSet.create(
            session_id=session_id,
            request=request,
        )
        variation_set.artifact_ids = artifact_ids
        variation_set.diversity_score = diversity_score
        variation_set.rankings = rankings or {}

        return self._storage.store_variation_set(variation_set)

    def get_variation_set(self, set_id: str) -> VariationSet | None:
        """Get a variation set by ID.

        Args:
            set_id: Variation set identifier.

        Returns:
            Variation set if found, None otherwise.
        """
        return self._storage.get_variation_set(set_id)

    def get_variation_artifacts(self, set_id: str) -> list[GenerationArtifact]:
        """Get all artifacts in a variation set.

        Args:
            set_id: Variation set identifier.

        Returns:
            List of artifacts in the set.
        """
        variation_set = self.get_variation_set(set_id)
        if not variation_set:
            return []

        artifacts = []
        for artifact_id in variation_set.artifact_ids:
            artifact = self.get_artifact(artifact_id)
            if artifact:
                artifacts.append(artifact)
        return artifacts

    # =========================================================================
    # Interaction Logging
    # =========================================================================

    def store_interaction(
        self,
        tool_name: str,
        request_params: dict[str, Any],
        session_id: str | None = None,
        artifact_id: str | None = None,
        response_summary: dict[str, Any] | None = None,
        feedback: str | None = None,
        agent_id: str | None = None,
    ) -> Interaction:
        """Log an MCP tool interaction.

        Args:
            tool_name: Name of the MCP tool called.
            request_params: Input parameters.
            session_id: Session to attach to (uses current if None).
            artifact_id: Linked artifact for generation tools.
            response_summary: Summary of response.
            feedback: For refine_layout, the feedback string.
            agent_id: Optional client/agent identifier.

        Returns:
            Stored interaction.
        """
        if session_id is None:
            session = self.get_or_create_session()
            session_id = session.id

        interaction = Interaction.create(
            session_id=session_id,
            tool_name=tool_name,
            request_params=request_params,
            artifact_id=artifact_id,
            response_summary=response_summary,
            feedback=feedback,
            agent_id=agent_id,
        )

        stored = self._storage.store_interaction(interaction)
        logger.debug(f"Logged interaction {stored.id}: {tool_name}")
        return stored

    def get_interaction(self, interaction_id: str) -> Interaction | None:
        """Get an interaction by ID.

        Args:
            interaction_id: Interaction identifier.

        Returns:
            Interaction if found, None otherwise.
        """
        return self._storage.get_interaction(interaction_id)

    def list_interactions(
        self,
        session_id: str | None = None,
        artifact_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Interaction]:
        """List interactions with filtering options.

        Args:
            session_id: Filter by session.
            artifact_id: Filter by linked artifact.
            tool_name: Filter by tool name.
            limit: Maximum interactions to return.
            offset: Number to skip.

        Returns:
            List of interactions ordered by created_at descending.
        """
        return self._storage.list_interactions(
            session_id=session_id,
            artifact_id=artifact_id,
            tool_name=tool_name,
            limit=limit,
            offset=offset,
        )

    def get_session_timeline(
        self,
        session_id: str,
        include_artifacts: bool = True,
    ) -> list[dict[str, Any]]:
        """Get chronological timeline of session activity.

        Combines interactions and artifacts into a unified timeline
        for conversation reconstruction.

        Args:
            session_id: Session to get timeline for.
            include_artifacts: Include artifact details.

        Returns:
            List of timeline events ordered by timestamp.
        """
        timeline: list[dict[str, Any]] = []

        # Get all interactions
        interactions = self.list_interactions(session_id=session_id, limit=1000)
        for interaction in interactions:
            event: dict[str, Any] = {
                "type": "interaction",
                "timestamp": interaction.created_at.isoformat(),
                "tool_name": interaction.tool_name,
                "request_params": interaction.request_params,
                "artifact_id": interaction.artifact_id,
                "feedback": interaction.feedback,
            }
            if interaction.response_summary:
                event["response_summary"] = interaction.response_summary
            timeline.append(event)

        # Optionally include artifacts
        if include_artifacts:
            artifacts = self.list_artifacts(session_id=session_id, limit=1000)
            for artifact in artifacts:
                event = {
                    "type": "artifact",
                    "timestamp": artifact.created_at.isoformat(),
                    "artifact_id": artifact.id,
                    "query": artifact.query,
                    "draft_preview": artifact.draft[:200]
                    if len(artifact.draft) > 200
                    else artifact.draft,
                    "parent_id": artifact.parent_id,
                    "model": artifact.model,
                }
                timeline.append(event)

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    # =========================================================================
    # Preview Cache
    # =========================================================================

    def cache_preview(
        self,
        artifact_id: str,
        style: str,
        image_bytes: bytes,
        format: str = "png",
    ) -> bool:
        """Cache a rendered preview for an artifact.

        Args:
            artifact_id: Artifact the preview is for.
            style: Preview style.
            image_bytes: Image data.
            format: Image format.

        Returns:
            True if cached.
        """
        return self._storage.cache_preview(artifact_id, style, image_bytes, format)

    def get_cached_preview(
        self,
        artifact_id: str,
        style: str,
    ) -> bytes | None:
        """Get a cached preview.

        Args:
            artifact_id: Artifact ID.
            style: Preview style.

        Returns:
            Image bytes if cached, None otherwise.
        """
        return self._storage.get_cached_preview(artifact_id, style)

    # =========================================================================
    # Storage Management
    # =========================================================================

    def get_stats(self) -> StorageStats:
        """Get storage statistics.

        Returns:
            Current storage statistics.
        """
        return self._storage.get_stats()

    def cleanup(
        self,
        delete_orphans: bool = True,
        delete_stale: bool = True,
        enforce_size_limit: bool = True,
        enforce_count_limit: bool = True,
    ) -> CleanupResult:
        """Perform storage cleanup.

        Removes orphaned artifacts, stale data, and enforces size limits
        by evicting least recently used items.

        Args:
            delete_orphans: Delete artifacts without sessions.
            delete_stale: Delete artifacts exceeding max age.
            enforce_size_limit: Enforce total storage size limit.
            enforce_count_limit: Enforce maximum artifact count.

        Returns:
            Cleanup results with counts and bytes freed.
        """
        return self._storage.cleanup(
            delete_orphans=delete_orphans,
            delete_stale=delete_stale,
            enforce_size_limit=enforce_size_limit,
            enforce_count_limit=enforce_count_limit,
        )

    def compact(self) -> CleanupResult:
        """Aggressively compact storage.

        Runs full cleanup with all options enabled.

        Returns:
            Cleanup results.
        """
        return self.cleanup(
            delete_orphans=True,
            delete_stale=True,
            enforce_size_limit=True,
            enforce_count_limit=True,
        )


# Global instance for convenience
_global_manager: HistoryManager | None = None


def get_history_manager(
    db_path: Path | str | None = None,
    config: StorageConfig | None = None,
) -> HistoryManager:
    """Get or create the global history manager.

    Args:
        db_path: Database path (only used on first call).
        config: Storage config (only used on first call).

    Returns:
        Global HistoryManager instance.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = HistoryManager(db_path=db_path, config=config)
    return _global_manager


def close_history_manager() -> None:
    """Close and clear the global history manager."""
    global _global_manager
    if _global_manager:
        _global_manager.close()
        _global_manager = None


__all__ = [
    "HistoryManager",
    "get_history_manager",
    "close_history_manager",
]
