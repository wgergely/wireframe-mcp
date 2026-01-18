"""SQLite storage backend for history management.

Provides persistent storage with size management, cleanup, and preview caching.
"""

import json
import logging
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..models import (
    ArtifactStatus,
    CleanupResult,
    GenerationArtifact,
    GenerationStats,
    Session,
    StorageConfig,
    StorageStats,
    VariationRequest,
    VariationSet,
)

logger = logging.getLogger(__name__)

# SQL Schema
SCHEMA_SQL = """
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    tags TEXT,  -- JSON array
    artifact_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    size_bytes INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL
);

-- Artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    parent_id TEXT,
    query TEXT NOT NULL,
    layout TEXT NOT NULL,  -- JSON
    draft TEXT NOT NULL,
    model TEXT,
    temperature REAL DEFAULT 0.7,
    provider TEXT DEFAULT 'd2',
    stats TEXT,  -- JSON
    rag_example_ids TEXT,  -- JSON array
    rag_scores TEXT,  -- JSON dict
    variation_group TEXT,
    variation_index INTEGER,
    tags TEXT,  -- JSON array
    status TEXT DEFAULT 'active',
    size_bytes INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES artifacts(id) ON DELETE SET NULL
);

-- Variation sets table
CREATE TABLE IF NOT EXISTS variation_sets (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    request TEXT NOT NULL,  -- JSON
    artifact_ids TEXT,  -- JSON array
    diversity_score REAL DEFAULT 0.0,
    rankings TEXT,  -- JSON dict
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_parent ON artifacts(parent_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_variation_group ON artifacts(variation_group);
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_accessed ON artifacts(accessed_at);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_variation_sets_session ON variation_sets(session_id);
"""


class SQLiteStorage:
    """SQLite-based storage backend with size management.

    Features:
    - Persistent storage in SQLite database
    - Automatic cleanup of old/orphaned artifacts
    - Size limit enforcement (LRU eviction)
    - Preview cache on filesystem

    Args:
        db_path: Path to SQLite database file.
        config: Storage configuration.
        preview_dir: Directory for preview cache (defaults to db_path.parent/previews).
    """

    def __init__(
        self,
        db_path: Path | str,
        config: StorageConfig | None = None,
        preview_dir: Path | str | None = None,
    ):
        self.db_path = Path(db_path)
        self._config = config or StorageConfig()
        self.preview_dir = (
            Path(preview_dir) if preview_dir else self.db_path.parent / "previews"
        )
        self._conn: sqlite3.Connection | None = None

    @property
    def config(self) -> StorageConfig:
        """Get storage configuration."""
        return self._config

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection, creating if needed."""
        if self._conn is None:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        return self._conn

    def initialize(self) -> None:
        """Initialize storage (create database, tables, directories)."""
        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._config.preview_cache_enabled:
            self.preview_dir.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")

        # Create schema
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

        logger.info(f"Initialized SQLite storage at {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # =========================================================================
    # Session Operations
    # =========================================================================

    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO sessions (
                id, name, description, tags, artifact_count,
                total_tokens, size_bytes, created_at, updated_at, accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.name,
                session.description,
                json.dumps(session.tags),
                session.artifact_count,
                session.total_tokens,
                session.size_bytes,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.accessed_at.isoformat(),
            ),
        )
        conn.commit()
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row:
            return self._row_to_session(row)
        return None

    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        tags: list[str] | None = None,
    ) -> list[Session]:
        """List sessions with optional filtering."""
        conn = self._get_conn()

        query = "SELECT * FROM sessions"
        params: list[Any] = []

        if tags:
            # Filter by any matching tag
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            query += f" WHERE ({tag_conditions})"
            params.extend([f'%"{tag}"%' for tag in tags])

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_session(row) for row in rows]

    def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        conn = self._get_conn()
        session.touch()
        conn.execute(
            """
            UPDATE sessions SET name = ?, description = ?, tags = ?,
                               artifact_count = ?, total_tokens = ?, size_bytes = ?,
                               updated_at = ?, accessed_at = ?
            WHERE id = ?
            """,
            (
                session.name,
                session.description,
                json.dumps(session.tags),
                session.artifact_count,
                session.total_tokens,
                session.size_bytes,
                session.updated_at.isoformat(),
                session.accessed_at.isoformat(),
                session.id,
            ),
        )
        conn.commit()
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts."""
        conn = self._get_conn()

        # Check if exists
        row = conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return False

        # Delete (CASCADE handles artifacts and variation_sets)
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

        # Clean up preview cache
        self._delete_session_previews(session_id)

        return True

    # =========================================================================
    # Artifact Operations
    # =========================================================================

    def store_artifact(self, artifact: GenerationArtifact) -> GenerationArtifact:
        """Store a new artifact."""
        conn = self._get_conn()

        # Ensure size is calculated
        if artifact.size_bytes == 0:
            artifact.size_bytes = artifact.estimate_size()

        conn.execute(
            """
            INSERT INTO artifacts (id, session_id, parent_id, query, layout, draft,
                                   model, temperature, provider, stats, rag_example_ids,
                                   rag_scores, variation_group, variation_index, tags,
                                   status, size_bytes, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.session_id,
                artifact.parent_id,
                artifact.query,
                json.dumps(artifact.layout),
                artifact.draft,
                artifact.model,
                artifact.temperature,
                artifact.provider,
                json.dumps(self._stats_to_dict(artifact.stats)),
                json.dumps(artifact.rag_example_ids),
                json.dumps(artifact.rag_scores),
                artifact.variation_group,
                artifact.variation_index,
                json.dumps(artifact.tags),
                artifact.status.value,
                artifact.size_bytes,
                artifact.created_at.isoformat(),
                artifact.accessed_at.isoformat(),
            ),
        )

        # Update session stats
        self._update_session_stats(artifact.session_id)

        conn.commit()
        return artifact

    def get_artifact(self, artifact_id: str) -> GenerationArtifact | None:
        """Get an artifact by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        if row:
            return self._row_to_artifact(row)
        return None

    def list_artifacts(
        self,
        session_id: str | None = None,
        parent_id: str | None = None,
        variation_group: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GenerationArtifact]:
        """List artifacts with filtering options."""
        conn = self._get_conn()

        query = "SELECT * FROM artifacts WHERE status = 'active'"
        params: list[Any] = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if parent_id:
            query += " AND parent_id = ?"
            params.append(parent_id)
        if variation_group:
            query += " AND variation_group = ?"
            params.append(variation_group)
        if tags:
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            query += f" AND ({tag_conditions})"
            params.extend([f'%"{tag}"%' for tag in tags])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def update_artifact(self, artifact: GenerationArtifact) -> GenerationArtifact:
        """Update an existing artifact."""
        conn = self._get_conn()
        artifact.touch()
        conn.execute(
            """
            UPDATE artifacts SET query = ?, layout = ?, draft = ?, model = ?,
                                temperature = ?, provider = ?, stats = ?,
                                rag_example_ids = ?, rag_scores = ?,
                                variation_group = ?, variation_index = ?,
                                tags = ?, status = ?, size_bytes = ?, accessed_at = ?
            WHERE id = ?
            """,
            (
                artifact.query,
                json.dumps(artifact.layout),
                artifact.draft,
                artifact.model,
                artifact.temperature,
                artifact.provider,
                json.dumps(self._stats_to_dict(artifact.stats)),
                json.dumps(artifact.rag_example_ids),
                json.dumps(artifact.rag_scores),
                artifact.variation_group,
                artifact.variation_index,
                json.dumps(artifact.tags),
                artifact.status.value,
                artifact.size_bytes,
                artifact.accessed_at.isoformat(),
                artifact.id,
            ),
        )
        conn.commit()
        return artifact

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        conn = self._get_conn()

        # Get artifact to find session
        row = conn.execute(
            "SELECT session_id, size_bytes FROM artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
        if not row:
            return False

        session_id = row["session_id"]

        # Delete artifact
        conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))

        # Update session stats
        self._update_session_stats(session_id)

        conn.commit()

        # Clean up preview cache
        self._delete_artifact_previews(artifact_id)

        return True

    def touch_artifact(self, artifact_id: str) -> bool:
        """Update artifact's accessed_at timestamp."""
        conn = self._get_conn()
        result = conn.execute(
            "UPDATE artifacts SET accessed_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), artifact_id),
        )
        conn.commit()
        return result.rowcount > 0

    # =========================================================================
    # Variation Set Operations
    # =========================================================================

    def store_variation_set(self, variation_set: VariationSet) -> VariationSet:
        """Store a new variation set."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO variation_sets (id, session_id, request, artifact_ids,
                                        diversity_score, rankings, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                variation_set.id,
                variation_set.session_id,
                json.dumps(self._request_to_dict(variation_set.request)),
                json.dumps(variation_set.artifact_ids),
                variation_set.diversity_score,
                json.dumps(variation_set.rankings),
                variation_set.created_at.isoformat(),
            ),
        )
        conn.commit()
        return variation_set

    def get_variation_set(self, set_id: str) -> VariationSet | None:
        """Get a variation set by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM variation_sets WHERE id = ?", (set_id,)
        ).fetchone()
        if row:
            return self._row_to_variation_set(row)
        return None

    def list_variation_sets(
        self,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[VariationSet]:
        """List variation sets."""
        conn = self._get_conn()

        query = "SELECT * FROM variation_sets"
        params: list[Any] = []

        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_variation_set(row) for row in rows]

    # =========================================================================
    # Query Operations
    # =========================================================================

    def search_artifacts_by_query(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[GenerationArtifact, float]]:
        """Search artifacts by query similarity (simple text match for now)."""
        conn = self._get_conn()

        # Simple LIKE search - could be enhanced with FTS5 or embedding search
        sql = """
            SELECT *, 1.0 as score FROM artifacts
            WHERE status = 'active' AND query LIKE ?
        """
        params: list[Any] = [f"%{query}%"]

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [(self._row_to_artifact(row), row["score"]) for row in rows]

    def get_artifact_children(self, artifact_id: str) -> list[GenerationArtifact]:
        """Get all artifacts that have this artifact as parent."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE parent_id = ? ORDER BY created_at",
            (artifact_id,),
        ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def get_artifact_lineage(
        self,
        artifact_id: str,
    ) -> dict[str, list[GenerationArtifact]]:
        """Get full lineage for an artifact."""
        conn = self._get_conn()

        # Get ancestors (walk up parent chain)
        ancestors = []
        current_id = artifact_id
        while True:
            row = conn.execute(
                "SELECT parent_id FROM artifacts WHERE id = ?", (current_id,)
            ).fetchone()
            if not row or not row["parent_id"]:
                break
            parent = self.get_artifact(row["parent_id"])
            if parent:
                ancestors.append(parent)
                current_id = parent.id
            else:
                break

        # Get descendants (recursive children)
        descendants = self._get_descendants_recursive(artifact_id)

        return {"ancestors": ancestors, "descendants": descendants}

    def _get_descendants_recursive(self, artifact_id: str) -> list[GenerationArtifact]:
        """Recursively get all descendants of an artifact."""
        children = self.get_artifact_children(artifact_id)
        descendants = list(children)
        for child in children:
            descendants.extend(self._get_descendants_recursive(child.id))
        return descendants

    # =========================================================================
    # Storage Management
    # =========================================================================

    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        conn = self._get_conn()

        # Get counts
        artifact_count = conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE status = 'active'"
        ).fetchone()[0]
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        variation_set_count = conn.execute(
            "SELECT COUNT(*) FROM variation_sets"
        ).fetchone()[0]

        # Get total size
        total_size = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM artifacts WHERE status = 'active'"
        ).fetchone()[0]

        # Get oldest artifact age
        oldest_row = conn.execute(
            "SELECT MIN(created_at) FROM artifacts WHERE status = 'active'"
        ).fetchone()
        oldest_artifact_age_days = 0.0
        if oldest_row and oldest_row[0]:
            oldest_date = datetime.fromisoformat(oldest_row[0])
            oldest_artifact_age_days = (datetime.now(UTC) - oldest_date).days

        # Get orphan count
        orphan_count = conn.execute(
            """
            SELECT COUNT(*) FROM artifacts
            WHERE session_id NOT IN (SELECT id FROM sessions)
            """
        ).fetchone()[0]

        # Get preview cache size
        preview_size = 0
        if self.preview_dir.exists():
            preview_size = sum(
                f.stat().st_size for f in self.preview_dir.rglob("*") if f.is_file()
            )

        return StorageStats(
            total_size_bytes=total_size,
            artifact_count=artifact_count,
            session_count=session_count,
            variation_set_count=variation_set_count,
            preview_cache_size_bytes=preview_size,
            oldest_artifact_age_days=oldest_artifact_age_days,
            orphan_count=orphan_count,
        )

    def find_orphans(self) -> list[GenerationArtifact]:
        """Find artifacts without valid sessions."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE session_id NOT IN (SELECT id FROM sessions)
            """
        ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def find_stale_artifacts(self, max_age_days: int) -> list[GenerationArtifact]:
        """Find artifacts older than max age."""
        conn = self._get_conn()
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE status = 'active' AND created_at < ?
            ORDER BY created_at ASC
            """,
            (cutoff.isoformat(),),
        ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def find_lru_artifacts(self, count: int) -> list[GenerationArtifact]:
        """Find least recently used artifacts."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE status = 'active'
            ORDER BY accessed_at ASC
            LIMIT ?
            """,
            (count,),
        ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def cleanup(
        self,
        delete_orphans: bool = True,
        delete_stale: bool = True,
        enforce_size_limit: bool = True,
        enforce_count_limit: bool = True,
    ) -> CleanupResult:
        """Perform storage cleanup."""
        result = CleanupResult()
        config = self._config

        try:
            # 1. Delete orphans
            if delete_orphans:
                orphans = self.find_orphans()
                for artifact in orphans:
                    if self.delete_artifact(artifact.id):
                        result.orphans_cleaned += 1
                        result.bytes_freed += artifact.size_bytes
                        result.artifacts_deleted += 1

            # 2. Delete stale artifacts
            if delete_stale:
                stale = self.find_stale_artifacts(config.max_artifact_age_days)
                for artifact in stale:
                    if self.delete_artifact(artifact.id):
                        result.bytes_freed += artifact.size_bytes
                        result.artifacts_deleted += 1

            # 3. Enforce count limit (delete oldest first)
            if enforce_count_limit:
                stats = self.get_stats()
                excess = stats.artifact_count - config.max_artifacts
                if excess > 0:
                    lru_artifacts = self.find_lru_artifacts(excess)
                    for artifact in lru_artifacts:
                        if self.delete_artifact(artifact.id):
                            result.bytes_freed += artifact.size_bytes
                            result.artifacts_deleted += 1

            # 4. Enforce size limit (delete LRU until under limit)
            if enforce_size_limit:
                stats = self.get_stats()
                while stats.total_size_bytes > config.max_total_size_bytes:
                    lru_artifacts = self.find_lru_artifacts(config.cleanup_batch_size)
                    if not lru_artifacts:
                        break
                    for artifact in lru_artifacts:
                        if self.delete_artifact(artifact.id):
                            result.bytes_freed += artifact.size_bytes
                            result.artifacts_deleted += 1
                    stats = self.get_stats()
                    if stats.total_size_bytes <= config.max_total_size_bytes:
                        break

            # 5. Clean up empty sessions
            self._delete_empty_sessions(result)

            # 6. Clean preview cache if over limit
            if config.preview_cache_enabled:
                self._cleanup_preview_cache(result)

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            result.errors.append(str(e))

        logger.info(
            f"Cleanup complete: {result.artifacts_deleted} artifacts deleted, "
            f"{result.mb_freed:.2f} MB freed"
        )
        return result

    def _delete_empty_sessions(self, result: CleanupResult) -> None:
        """Delete sessions with no artifacts."""
        conn = self._get_conn()
        empty_sessions = conn.execute(
            """
            SELECT id FROM sessions
            WHERE id NOT IN (SELECT DISTINCT session_id FROM artifacts)
            """
        ).fetchall()
        for row in empty_sessions:
            if self.delete_session(row["id"]):
                result.sessions_deleted += 1

    def _cleanup_preview_cache(self, result: CleanupResult) -> None:
        """Clean preview cache if over limit."""
        if not self.preview_dir.exists():
            return

        # Get current size
        current_size = sum(
            f.stat().st_size for f in self.preview_dir.rglob("*") if f.is_file()
        )

        if current_size <= self._config.preview_max_size_bytes:
            return

        # Delete oldest previews first
        preview_files = sorted(
            self.preview_dir.rglob("*"),
            key=lambda f: f.stat().st_mtime if f.is_file() else 0,
        )

        for f in preview_files:
            if not f.is_file():
                continue
            if current_size <= self._config.preview_max_size_bytes:
                break
            size = f.stat().st_size
            f.unlink()
            current_size -= size
            result.bytes_freed += size
            result.previews_deleted += 1

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
        """Cache a rendered preview."""
        if not self._config.preview_cache_enabled:
            return False

        preview_path = self.preview_dir / artifact_id
        preview_path.mkdir(parents=True, exist_ok=True)

        file_path = preview_path / f"{style}.{format}"
        file_path.write_bytes(image_bytes)
        return True

    def get_cached_preview(
        self,
        artifact_id: str,
        style: str,
    ) -> bytes | None:
        """Get a cached preview."""
        if not self._config.preview_cache_enabled:
            return None

        preview_path = self.preview_dir / artifact_id
        for ext in ["png", "svg"]:
            file_path = preview_path / f"{style}.{ext}"
            if file_path.exists():
                return file_path.read_bytes()
        return None

    def clear_preview_cache(self, artifact_id: str | None = None) -> int:
        """Clear preview cache."""
        count = 0
        if artifact_id:
            preview_path = self.preview_dir / artifact_id
            if preview_path.exists():
                count = len(list(preview_path.iterdir()))
                shutil.rmtree(preview_path)
        else:
            if self.preview_dir.exists():
                count = sum(1 for _ in self.preview_dir.rglob("*") if _.is_file())
                shutil.rmtree(self.preview_dir)
                self.preview_dir.mkdir(parents=True, exist_ok=True)
        return count

    def _delete_artifact_previews(self, artifact_id: str) -> None:
        """Delete previews for an artifact."""
        preview_path = self.preview_dir / artifact_id
        if preview_path.exists():
            shutil.rmtree(preview_path)

    def _delete_session_previews(self, session_id: str) -> None:
        """Delete all previews for artifacts in a session."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM artifacts WHERE session_id = ?", (session_id,)
        ).fetchall()
        for row in rows:
            self._delete_artifact_previews(row["id"])

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _update_session_stats(self, session_id: str) -> None:
        """Update session statistics from artifacts."""
        conn = self._get_conn()

        stats = conn.execute(
            """
            SELECT COUNT(*) as count,
                   COALESCE(SUM(size_bytes), 0) as size,
                   COALESCE(SUM(json_extract(stats, '$.total_tokens')), 0) as tokens
            FROM artifacts
            WHERE session_id = ? AND status = 'active'
            """,
            (session_id,),
        ).fetchone()

        conn.execute(
            """
            UPDATE sessions SET artifact_count = ?, size_bytes = ?,
                               total_tokens = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                stats["count"],
                stats["size"],
                stats["tokens"],
                datetime.now(UTC).isoformat(),
                session_id,
            ),
        )

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert database row to Session object."""
        return Session(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            artifact_count=row["artifact_count"],
            total_tokens=row["total_tokens"],
            size_bytes=row["size_bytes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
        )

    def _row_to_artifact(self, row: sqlite3.Row) -> GenerationArtifact:
        """Convert database row to GenerationArtifact object."""
        stats_dict = json.loads(row["stats"]) if row["stats"] else {}
        return GenerationArtifact(
            id=row["id"],
            session_id=row["session_id"],
            parent_id=row["parent_id"],
            query=row["query"],
            layout=json.loads(row["layout"]),
            draft=row["draft"],
            model=row["model"] or "",
            temperature=row["temperature"],
            provider=row["provider"] or "d2",
            stats=GenerationStats(**stats_dict) if stats_dict else GenerationStats(),
            rag_example_ids=json.loads(row["rag_example_ids"])
            if row["rag_example_ids"]
            else [],
            rag_scores=json.loads(row["rag_scores"]) if row["rag_scores"] else {},
            variation_group=row["variation_group"],
            variation_index=row["variation_index"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            status=ArtifactStatus(row["status"]),
            size_bytes=row["size_bytes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
        )

    def _row_to_variation_set(self, row: sqlite3.Row) -> VariationSet:
        """Convert database row to VariationSet object."""
        request_dict = json.loads(row["request"])
        return VariationSet(
            id=row["id"],
            session_id=row["session_id"],
            request=VariationRequest(**request_dict),
            artifact_ids=json.loads(row["artifact_ids"]) if row["artifact_ids"] else [],
            diversity_score=row["diversity_score"],
            rankings=json.loads(row["rankings"]) if row["rankings"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _stats_to_dict(self, stats: GenerationStats) -> dict[str, Any]:
        """Convert GenerationStats to dict."""
        return {
            "attempts": stats.attempts,
            "validation_retries": stats.validation_retries,
            "json_repairs": stats.json_repairs,
            "total_tokens": stats.total_tokens,
            "final_model": stats.final_model,
            "generation_time_ms": stats.generation_time_ms,
        }

    def _request_to_dict(self, request: VariationRequest) -> dict[str, Any]:
        """Convert VariationRequest to dict."""
        return {
            "query": request.query,
            "count": request.count,
            "base_artifact_id": request.base_artifact_id,
            "temperature_range": request.temperature_range,
            "model": request.model,
            "include_rag": request.include_rag,
        }


__all__ = ["SQLiteStorage"]
