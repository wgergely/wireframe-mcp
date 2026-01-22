"""Data models for history management.

This module defines the core data structures for artifact persistence,
session management, and variation tracking.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class ArtifactStatus(str, Enum):
    """Status of a generation artifact."""

    ACTIVE = "active"  # Normal, accessible artifact
    ARCHIVED = "archived"  # Marked for potential cleanup
    ORPHANED = "orphaned"  # No session, will be cleaned up


@dataclass
class GenerationStats:
    """Statistics from a generation operation."""

    attempts: int = 1
    validation_retries: int = 0
    json_repairs: int = 0
    total_tokens: int = 0
    final_model: str = ""
    generation_time_ms: int = 0


@dataclass
class GenerationArtifact:
    """A persisted generated layout with full metadata.

    Attributes:
        id: Unique identifier for this artifact.
        session_id: Parent session grouping related operations.
        parent_id: For refinements/variations, links to source artifact.
        layout: The actual layout structure as JSON-serializable dict.
        draft: Text tree representation for quick review.
        query: Original natural language query.
        model: LLM model used for generation.
        temperature: Generation temperature setting.
        provider: DSL provider hint used.
        stats: Generation statistics (attempts, tokens, etc.).
        rag_example_ids: IDs of corpus items used for context.
        rag_scores: Relevance scores for each RAG example.
        variation_group: Groups related variations together.
        variation_index: Position in variation set (0-indexed).
        tags: User-defined tags for organization.
        status: Artifact lifecycle status.
        size_bytes: Approximate storage size.
        created_at: Creation timestamp.
        accessed_at: Last access timestamp (for LRU eviction).
    """

    id: str
    session_id: str
    query: str
    layout: dict[str, Any]
    draft: str

    # Generation context
    model: str = ""
    temperature: float = 0.7
    provider: str = "d2"
    stats: GenerationStats = field(default_factory=GenerationStats)

    # RAG lineage
    rag_example_ids: list[str] = field(default_factory=list)
    rag_scores: dict[str, float] = field(default_factory=dict)

    # Relationships
    parent_id: str | None = None
    variation_group: str | None = None
    variation_index: int | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    status: ArtifactStatus = ArtifactStatus.ACTIVE
    size_bytes: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        session_id: str,
        query: str,
        layout: dict[str, Any],
        draft: str,
        **kwargs: Any,
    ) -> "GenerationArtifact":
        """Factory method to create a new artifact with generated ID."""
        artifact = cls(
            id=str(uuid4()),
            session_id=session_id,
            query=query,
            layout=layout,
            draft=draft,
            **kwargs,
        )
        artifact.size_bytes = artifact.estimate_size()
        return artifact

    def estimate_size(self) -> int:
        """Estimate storage size in bytes."""
        import json

        # Layout JSON size
        layout_size = len(json.dumps(self.layout))
        # Draft text size
        draft_size = len(self.draft.encode("utf-8"))
        # Query size
        query_size = len(self.query.encode("utf-8"))
        # Metadata overhead (approximate)
        metadata_size = 500

        return layout_size + draft_size + query_size + metadata_size

    def touch(self) -> None:
        """Update accessed_at timestamp."""
        self.accessed_at = datetime.now(UTC)


@dataclass
class Session:
    """A logical grouping of related generation operations.

    Attributes:
        id: Unique session identifier.
        name: Optional user-friendly name.
        description: Optional description.
        tags: User-defined tags.
        artifact_count: Number of artifacts in session.
        total_tokens: Total tokens used across all artifacts.
        size_bytes: Total storage size of session.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
        accessed_at: Last access timestamp.
    """

    id: str
    name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    # Statistics
    artifact_count: int = 0
    total_tokens: int = 0
    size_bytes: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> "Session":
        """Factory method to create a new session with generated ID."""
        return cls(
            id=str(uuid4()),
            name=name,
            description=description,
            tags=tags or [],
        )

    def touch(self) -> None:
        """Update accessed_at and updated_at timestamps."""
        now = datetime.now(UTC)
        self.accessed_at = now
        self.updated_at = now


@dataclass
class VariationRequest:
    """Request structure for generating multiple variations.

    Attributes:
        query: Base natural language query.
        count: Number of variations to generate (1-10).
        base_artifact_id: Generate variations of existing artifact.
        temperature_range: Min/max temperature spread.
        model: LLM model to use.
        include_rag: Whether to include RAG context.
    """

    query: str
    count: int = 3
    base_artifact_id: str | None = None
    temperature_range: tuple[float, float] | None = None
    model: str | None = None
    include_rag: bool = True

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not 1 <= self.count <= 10:
            raise ValueError(f"count must be 1-10, got {self.count}")
        if self.temperature_range:
            low, high = self.temperature_range
            if not (0.0 <= low <= high <= 2.0):
                raise ValueError(f"Invalid temperature_range: {self.temperature_range}")


@dataclass
class VariationSet:
    """A set of related layout variations.

    Attributes:
        id: Unique identifier for this set.
        session_id: Parent session.
        request: Original variation request.
        artifact_ids: IDs of generated artifacts.
        diversity_score: How different the variations are (0-1).
        rankings: Criterion -> ordered artifact IDs.
        created_at: Creation timestamp.
    """

    id: str
    session_id: str
    request: VariationRequest
    artifact_ids: list[str] = field(default_factory=list)
    diversity_score: float = 0.0
    rankings: dict[str, list[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        session_id: str,
        request: VariationRequest,
    ) -> "VariationSet":
        """Factory method to create a new variation set."""
        return cls(
            id=str(uuid4()),
            session_id=session_id,
            request=request,
        )


@dataclass
class StorageConfig:
    """Configuration for history storage with size management.

    Attributes:
        max_total_size_mb: Maximum total storage size in MB.
        max_artifacts: Maximum number of artifacts to keep.
        max_artifact_age_days: Maximum age of artifacts in days.
        max_session_artifacts: Maximum artifacts per session.
        cleanup_batch_size: Number of items to clean per batch.
        orphan_grace_period_hours: Hours before orphaned artifacts are deleted.
        preview_cache_enabled: Whether to cache rendered previews.
        preview_max_size_mb: Maximum preview cache size in MB.
    """

    max_total_size_mb: int = 500  # 500 MB default
    max_artifacts: int = 10000  # 10k artifacts max
    max_artifact_age_days: int = 90  # 90 days retention
    max_session_artifacts: int = 500  # 500 per session
    cleanup_batch_size: int = 100  # Clean 100 at a time
    orphan_grace_period_hours: int = 24  # 24 hours grace
    preview_cache_enabled: bool = True
    preview_max_size_mb: int = 100  # 100 MB for previews

    @property
    def max_total_size_bytes(self) -> int:
        """Get max size in bytes."""
        return self.max_total_size_mb * 1024 * 1024

    @property
    def preview_max_size_bytes(self) -> int:
        """Get preview max size in bytes."""
        return self.preview_max_size_mb * 1024 * 1024


@dataclass
class StorageStats:
    """Statistics about storage usage.

    Attributes:
        total_size_bytes: Total storage used.
        artifact_count: Number of artifacts.
        session_count: Number of sessions.
        variation_set_count: Number of variation sets.
        preview_cache_size_bytes: Preview cache size.
        oldest_artifact_age_days: Age of oldest artifact.
        orphan_count: Number of orphaned artifacts.
    """

    total_size_bytes: int = 0
    artifact_count: int = 0
    session_count: int = 0
    variation_set_count: int = 0
    preview_cache_size_bytes: int = 0
    oldest_artifact_age_days: float = 0.0
    orphan_count: int = 0

    @property
    def total_size_mb(self) -> float:
        """Get total size in MB."""
        return self.total_size_bytes / (1024 * 1024)

    @property
    def preview_cache_size_mb(self) -> float:
        """Get preview cache size in MB."""
        return self.preview_cache_size_bytes / (1024 * 1024)


@dataclass
class CleanupResult:
    """Result of a storage cleanup operation.

    Attributes:
        artifacts_deleted: Number of artifacts deleted.
        sessions_deleted: Number of sessions deleted.
        previews_deleted: Number of preview files deleted.
        bytes_freed: Total bytes freed.
        orphans_cleaned: Number of orphans cleaned.
        errors: List of error messages.
    """

    artifacts_deleted: int = 0
    sessions_deleted: int = 0
    previews_deleted: int = 0
    bytes_freed: int = 0
    orphans_cleaned: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def mb_freed(self) -> float:
        """Get MB freed."""
        return self.bytes_freed / (1024 * 1024)


@dataclass
class Interaction:
    """A logged MCP tool interaction.

    Captures every tool call for conversation reconstruction
    and agent behavior auditing.

    Attributes:
        id: Unique interaction identifier.
        session_id: Parent session.
        artifact_id: Linked artifact (for generation tools).
        tool_name: MCP tool that was called.
        request_params: Input parameters as dict.
        response_summary: Summary of response (not full response).
        feedback: For refine_layout, the feedback string.
        agent_id: Optional client/agent identifier.
        created_at: Timestamp of the call.
    """

    id: str
    session_id: str
    tool_name: str
    request_params: dict[str, Any]

    artifact_id: str | None = None
    response_summary: dict[str, Any] | None = None
    feedback: str | None = None
    agent_id: str | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        session_id: str,
        tool_name: str,
        request_params: dict[str, Any],
        artifact_id: str | None = None,
        response_summary: dict[str, Any] | None = None,
        feedback: str | None = None,
        agent_id: str | None = None,
    ) -> "Interaction":
        """Factory method to create a new interaction with generated ID."""
        return cls(
            id=str(uuid4()),
            session_id=session_id,
            tool_name=tool_name,
            request_params=request_params,
            artifact_id=artifact_id,
            response_summary=response_summary,
            feedback=feedback,
            agent_id=agent_id,
        )


__all__ = [
    "ArtifactStatus",
    "GenerationStats",
    "GenerationArtifact",
    "Interaction",
    "Session",
    "VariationRequest",
    "VariationSet",
    "StorageConfig",
    "StorageStats",
    "CleanupResult",
]
