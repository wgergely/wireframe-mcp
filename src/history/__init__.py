"""History management for wireframe-mcp.

This module provides artifact persistence, session management,
variation generation, and lineage tracking for generated layouts.

Example:
    >>> from src.history import get_history_manager
    >>> manager = get_history_manager()
    >>> session = manager.get_or_create_session("my-project")
    >>> artifact = manager.store_artifact(
    ...     session_id=session.id,
    ...     query="login form with email and password",
    ...     layout={"type": "container", "children": [...]},
    ...     draft="Login [container]\\n├── Email [input]\\n...",
    ... )
    >>> print(f"Stored: {artifact.id}")

Features:
    - Artifact persistence with SQLite backend
    - Session management for grouping operations
    - Variation generation with diversity metrics
    - Lineage tracking (parent/child relationships)
    - Automatic storage size management
    - Preview caching

Configuration:
    Storage limits can be configured via StorageConfig:
    - max_total_size_mb: Maximum storage size (default: 500 MB)
    - max_artifacts: Maximum artifact count (default: 10,000)
    - max_artifact_age_days: Maximum age (default: 90 days)
"""

from .lib import (
    HistoryManager,
    close_history_manager,
    get_history_manager,
)
from .models import (
    ArtifactStatus,
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
from .variations import (
    calculate_structural_diversity,
    compare_artifacts,
    generate_rankings,
    generate_variations_sync,
)

__all__ = [
    # Manager
    "HistoryManager",
    "get_history_manager",
    "close_history_manager",
    # Models
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
    # Variations
    "calculate_structural_diversity",
    "compare_artifacts",
    "generate_rankings",
    "generate_variations_sync",
]
