"""Storage protocol for history management.

Defines the interface that all storage backends must implement.
"""

from typing import Protocol

from ..models import (
    CleanupResult,
    GenerationArtifact,
    Interaction,
    Session,
    StorageConfig,
    StorageStats,
    VariationSet,
)


class HistoryStorage(Protocol):
    """Protocol defining the storage interface for history management.

    All storage backends (SQLite, PostgreSQL, in-memory) must implement
    this interface to be compatible with HistoryManager.
    """

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def initialize(self) -> None:
        """Initialize storage (create tables, indexes, etc.)."""
        ...

    def close(self) -> None:
        """Close storage connections and clean up resources."""
        ...

    @property
    def config(self) -> StorageConfig:
        """Get storage configuration."""
        ...

    # =========================================================================
    # Session Operations
    # =========================================================================

    def create_session(self, session: Session) -> Session:
        """Create a new session.

        Args:
            session: Session to create.

        Returns:
            Created session with any server-generated fields.
        """
        ...

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session if found, None otherwise.
        """
        ...

    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        tags: list[str] | None = None,
    ) -> list[Session]:
        """List sessions with optional filtering.

        Args:
            limit: Maximum sessions to return.
            offset: Number of sessions to skip.
            tags: Filter by tags (any match).

        Returns:
            List of sessions ordered by updated_at descending.
        """
        ...

    def update_session(self, session: Session) -> Session:
        """Update an existing session.

        Args:
            session: Session with updated fields.

        Returns:
            Updated session.
        """
        ...

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts.

        Args:
            session_id: Session to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    # =========================================================================
    # Artifact Operations
    # =========================================================================

    def store_artifact(self, artifact: GenerationArtifact) -> GenerationArtifact:
        """Store a new artifact.

        Args:
            artifact: Artifact to store.

        Returns:
            Stored artifact with any server-generated fields.
        """
        ...

    def get_artifact(self, artifact_id: str) -> GenerationArtifact | None:
        """Get an artifact by ID.

        Args:
            artifact_id: Artifact identifier.

        Returns:
            Artifact if found, None otherwise.
        """
        ...

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
            tags: Filter by tags (any match).
            limit: Maximum artifacts to return.
            offset: Number of artifacts to skip.

        Returns:
            List of artifacts ordered by created_at descending.
        """
        ...

    def update_artifact(self, artifact: GenerationArtifact) -> GenerationArtifact:
        """Update an existing artifact.

        Args:
            artifact: Artifact with updated fields.

        Returns:
            Updated artifact.
        """
        ...

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: Artifact to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def touch_artifact(self, artifact_id: str) -> bool:
        """Update artifact's accessed_at timestamp.

        Args:
            artifact_id: Artifact to touch.

        Returns:
            True if updated, False if not found.
        """
        ...

    # =========================================================================
    # Variation Set Operations
    # =========================================================================

    def store_variation_set(self, variation_set: VariationSet) -> VariationSet:
        """Store a new variation set.

        Args:
            variation_set: Variation set to store.

        Returns:
            Stored variation set.
        """
        ...

    def get_variation_set(self, set_id: str) -> VariationSet | None:
        """Get a variation set by ID.

        Args:
            set_id: Variation set identifier.

        Returns:
            Variation set if found, None otherwise.
        """
        ...

    def list_variation_sets(
        self,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[VariationSet]:
        """List variation sets.

        Args:
            session_id: Filter by session.
            limit: Maximum sets to return.
            offset: Number of sets to skip.

        Returns:
            List of variation sets ordered by created_at descending.
        """
        ...

    # =========================================================================
    # Interaction Operations
    # =========================================================================

    def store_interaction(self, interaction: Interaction) -> Interaction:
        """Store a new interaction.

        Args:
            interaction: Interaction to store.

        Returns:
            Stored interaction.
        """
        ...

    def get_interaction(self, interaction_id: str) -> Interaction | None:
        """Get an interaction by ID.

        Args:
            interaction_id: Interaction identifier.

        Returns:
            Interaction if found, None otherwise.
        """
        ...

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
        ...

    # =========================================================================
    # Query Operations
    # =========================================================================

    def search_artifacts_by_query(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[GenerationArtifact, float]]:
        """Search artifacts by query similarity.

        Args:
            query: Search query text.
            session_id: Limit search to session.
            limit: Maximum results.

        Returns:
            List of (artifact, score) tuples ordered by relevance.
        """
        ...

    def get_artifact_children(self, artifact_id: str) -> list[GenerationArtifact]:
        """Get all artifacts that have this artifact as parent.

        Args:
            artifact_id: Parent artifact ID.

        Returns:
            List of child artifacts.
        """
        ...

    def get_artifact_lineage(
        self,
        artifact_id: str,
    ) -> dict[str, list[GenerationArtifact]]:
        """Get full lineage (ancestors and descendants) for an artifact.

        Args:
            artifact_id: Artifact to trace.

        Returns:
            Dict with 'ancestors' and 'descendants' lists.
        """
        ...

    # =========================================================================
    # Storage Management
    # =========================================================================

    def get_stats(self) -> StorageStats:
        """Get storage statistics.

        Returns:
            Current storage statistics.
        """
        ...

    def find_orphans(self) -> list[GenerationArtifact]:
        """Find artifacts without valid sessions.

        Returns:
            List of orphaned artifacts.
        """
        ...

    def find_stale_artifacts(
        self,
        max_age_days: int,
    ) -> list[GenerationArtifact]:
        """Find artifacts older than max age.

        Args:
            max_age_days: Maximum age in days.

        Returns:
            List of stale artifacts.
        """
        ...

    def find_lru_artifacts(
        self,
        count: int,
    ) -> list[GenerationArtifact]:
        """Find least recently used artifacts.

        Args:
            count: Number of artifacts to find.

        Returns:
            List of LRU artifacts ordered by accessed_at ascending.
        """
        ...

    def cleanup(
        self,
        delete_orphans: bool = True,
        delete_stale: bool = True,
        enforce_size_limit: bool = True,
        enforce_count_limit: bool = True,
    ) -> CleanupResult:
        """Perform storage cleanup.

        Args:
            delete_orphans: Delete orphaned artifacts.
            delete_stale: Delete artifacts exceeding max age.
            enforce_size_limit: Delete LRU artifacts to stay under size limit.
            enforce_count_limit: Delete LRU artifacts to stay under count limit.

        Returns:
            Cleanup operation results.
        """
        ...

    # =========================================================================
    # Preview Cache (Optional)
    # =========================================================================

    def cache_preview(
        self,
        artifact_id: str,
        style: str,
        image_bytes: bytes,
        format: str = "png",
    ) -> bool:
        """Cache a rendered preview.

        Args:
            artifact_id: Artifact the preview is for.
            style: Preview style (wireframe, sketch, minimal).
            image_bytes: Image data.
            format: Image format.

        Returns:
            True if cached successfully.
        """
        ...

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
        ...

    def clear_preview_cache(self, artifact_id: str | None = None) -> int:
        """Clear preview cache.

        Args:
            artifact_id: Clear only for this artifact, or all if None.

        Returns:
            Number of previews cleared.
        """
        ...


__all__ = ["HistoryStorage"]
