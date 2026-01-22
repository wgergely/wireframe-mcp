# Interaction Logging Tier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a structured interaction logging tier that enables LLM agents to reconstruct conversation context, audit MCP tool calls, and retrieve refinement feedback history.

**Architecture:** Add `Interaction` model and `interactions` table to capture every MCP tool call with parameters, link to artifacts, and store refinement feedback separately. Extend `HistoryManager` and add new MCP tools for session history retrieval. Add CLI `history` subcommand for management.

**Tech Stack:** Python 3.12, SQLite, FastMCP, Pydantic dataclasses

---

## Phase 1: Data Model & Schema

### Task 1.1: Add Interaction Model

**Files:**
- Modify: `src/history/models.py`

**Step 1: Write the failing test**

Add to `src/history/test.py`:

```python
class TestInteractionModel:
    """Tests for Interaction model."""

    def test_interaction_creation_via_factory(self):
        """Test Interaction model creation via factory."""
        from src.history.models import Interaction

        interaction = Interaction.create(
            session_id="test-session",
            tool_name="generate_layout",
            request_params={"query": "login form", "temperature": 0.7},
        )
        assert interaction.id is not None
        assert interaction.session_id == "test-session"
        assert interaction.tool_name == "generate_layout"
        assert interaction.request_params["query"] == "login form"
        assert interaction.artifact_id is None
        assert interaction.feedback is None
        assert interaction.agent_id is None

    def test_interaction_with_artifact_link(self):
        """Test Interaction with artifact link."""
        from src.history.models import Interaction

        interaction = Interaction.create(
            session_id="test-session",
            tool_name="generate_layout",
            request_params={"query": "dashboard"},
            artifact_id="artifact-123",
            response_summary={"status": "success", "node_count": 5},
        )
        assert interaction.artifact_id == "artifact-123"
        assert interaction.response_summary["status"] == "success"

    def test_interaction_with_feedback(self):
        """Test Interaction for refine_layout with feedback."""
        from src.history.models import Interaction

        interaction = Interaction.create(
            session_id="test-session",
            tool_name="refine_layout",
            request_params={"artifact_id": "parent-123"},
            artifact_id="child-456",
            feedback="make the sidebar narrower",
        )
        assert interaction.tool_name == "refine_layout"
        assert interaction.feedback == "make the sidebar narrower"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/history/test.py::TestInteractionModel -v`
Expected: FAIL with "cannot import name 'Interaction'"

**Step 3: Write minimal implementation**

Add to `src/history/models.py` before `__all__`:

```python
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
```

Update `__all__` in `src/history/models.py`:

```python
__all__ = [
    "ArtifactStatus",
    "GenerationStats",
    "GenerationArtifact",
    "Session",
    "VariationRequest",
    "VariationSet",
    "StorageConfig",
    "StorageStats",
    "CleanupResult",
    "Interaction",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest src/history/test.py::TestInteractionModel -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/history/models.py src/history/test.py
git commit -m "feat(history): add Interaction model for MCP tool logging"
```

---

### Task 1.2: Add Interactions Table to SQLite Schema

**Files:**
- Modify: `src/history/storage/sqlite.py`

**Step 1: Write the failing test**

Add to `src/history/test.py`:

```python
class TestInteractionStorage:
    """Tests for interaction storage operations."""

    def test_store_interaction(self, manager):
        """Test storing an interaction."""
        from src.history.models import Interaction

        session = manager.get_or_create_session("test-session")
        interaction = Interaction.create(
            session_id=session.id,
            tool_name="generate_layout",
            request_params={"query": "login form"},
        )

        stored = manager.store_interaction(interaction)
        assert stored.id == interaction.id
        assert stored.tool_name == "generate_layout"

    def test_get_interaction(self, manager):
        """Test retrieving an interaction."""
        from src.history.models import Interaction

        session = manager.get_or_create_session("test-session")
        interaction = Interaction.create(
            session_id=session.id,
            tool_name="status",
            request_params={},
        )
        manager.store_interaction(interaction)

        retrieved = manager.get_interaction(interaction.id)
        assert retrieved is not None
        assert retrieved.id == interaction.id

    def test_list_interactions_by_session(self, manager):
        """Test listing interactions for a session."""
        from src.history.models import Interaction

        session = manager.get_or_create_session("test-session")

        for i in range(5):
            interaction = Interaction.create(
                session_id=session.id,
                tool_name=f"tool_{i}",
                request_params={"index": i},
            )
            manager.store_interaction(interaction)

        interactions = manager.list_interactions(session_id=session.id)
        assert len(interactions) == 5

    def test_list_interactions_by_artifact(self, manager, sample_layout, sample_stats):
        """Test listing interactions linked to an artifact."""
        from src.history.models import Interaction

        session = manager.get_or_create_session("test-session")
        artifact = manager.store_artifact(
            query="test",
            layout=sample_layout,
            draft="Test",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        interaction = Interaction.create(
            session_id=session.id,
            tool_name="generate_layout",
            request_params={"query": "test"},
            artifact_id=artifact.id,
        )
        manager.store_interaction(interaction)

        interactions = manager.list_interactions(artifact_id=artifact.id)
        assert len(interactions) == 1
        assert interactions[0].artifact_id == artifact.id
```

**Step 2: Run test to verify it fails**

Run: `pytest src/history/test.py::TestInteractionStorage -v`
Expected: FAIL with "HistoryManager has no attribute 'store_interaction'"

**Step 3: Write minimal implementation**

Add to `SCHEMA_SQL` in `src/history/storage/sqlite.py` (after variation_sets table):

```python
# In SCHEMA_SQL string, add:
-- Interactions table
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    artifact_id TEXT,
    tool_name TEXT NOT NULL,
    request_params TEXT NOT NULL,  -- JSON
    response_summary TEXT,  -- JSON
    feedback TEXT,
    agent_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_artifact ON interactions(artifact_id);
CREATE INDEX IF NOT EXISTS idx_interactions_tool ON interactions(tool_name);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);
```

Add import to `src/history/storage/sqlite.py`:

```python
from ..models import (
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
```

Add methods to `SQLiteStorage` class:

```python
# =========================================================================
# Interaction Operations
# =========================================================================

def store_interaction(self, interaction: Interaction) -> Interaction:
    """Store a new interaction."""
    conn = self._get_conn()
    conn.execute(
        """
        INSERT INTO interactions (
            id, session_id, artifact_id, tool_name, request_params,
            response_summary, feedback, agent_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            interaction.id,
            interaction.session_id,
            interaction.artifact_id,
            interaction.tool_name,
            json.dumps(interaction.request_params),
            json.dumps(interaction.response_summary) if interaction.response_summary else None,
            interaction.feedback,
            interaction.agent_id,
            interaction.created_at.isoformat(),
        ),
    )
    conn.commit()
    return interaction

def get_interaction(self, interaction_id: str) -> Interaction | None:
    """Get an interaction by ID."""
    conn = self._get_conn()
    row = conn.execute(
        "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
    ).fetchone()
    if row:
        return self._row_to_interaction(row)
    return None

def list_interactions(
    self,
    session_id: str | None = None,
    artifact_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Interaction]:
    """List interactions with filtering options."""
    conn = self._get_conn()

    query = "SELECT * FROM interactions WHERE 1=1"
    params: list[Any] = []

    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if artifact_id:
        query += " AND artifact_id = ?"
        params.append(artifact_id)
    if tool_name:
        query += " AND tool_name = ?"
        params.append(tool_name)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    return [self._row_to_interaction(row) for row in rows]

def _row_to_interaction(self, row: sqlite3.Row) -> Interaction:
    """Convert database row to Interaction object."""
    return Interaction(
        id=row["id"],
        session_id=row["session_id"],
        artifact_id=row["artifact_id"],
        tool_name=row["tool_name"],
        request_params=json.loads(row["request_params"]),
        response_summary=json.loads(row["response_summary"]) if row["response_summary"] else None,
        feedback=row["feedback"],
        agent_id=row["agent_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest src/history/test.py::TestInteractionStorage -v`
Expected: Still FAIL - need to add to HistoryManager

---

### Task 1.3: Add Interaction Methods to HistoryManager

**Files:**
- Modify: `src/history/lib.py`

**Step 1: (Tests already exist from Task 1.2)**

**Step 2: Write implementation**

Add import to `src/history/lib.py`:

```python
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
```

Add methods to `HistoryManager` class (after variation set methods):

```python
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
                "draft_preview": artifact.draft[:200] if len(artifact.draft) > 200 else artifact.draft,
                "parent_id": artifact.parent_id,
                "model": artifact.model,
            }
            timeline.append(event)

    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"])

    return timeline
```

**Step 3: Run test to verify it passes**

Run: `pytest src/history/test.py::TestInteractionStorage -v`
Expected: PASS (4 tests)

**Step 4: Commit**

```bash
git add src/history/storage/sqlite.py src/history/lib.py src/history/test.py
git commit -m "feat(history): add interaction storage and timeline reconstruction"
```

---

### Task 1.4: Update Protocol and Exports

**Files:**
- Modify: `src/history/storage/protocol.py`
- Modify: `src/history/__init__.py`

**Step 1: Update protocol**

Add to `src/history/storage/protocol.py` imports:

```python
from ..models import (
    CleanupResult,
    GenerationArtifact,
    Interaction,
    Session,
    StorageConfig,
    StorageStats,
    VariationSet,
)
```

Add to `HistoryStorage` protocol class:

```python
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
```

**Step 2: Update exports**

Update `src/history/__init__.py`:

```python
"""History management for wireframe-mcp.

This module provides artifact persistence, session management,
variation generation, interaction logging, and lineage tracking
for generated layouts.

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
    - Interaction logging for conversation reconstruction
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
```

**Step 3: Run all history tests**

Run: `pytest src/history/test.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/history/storage/protocol.py src/history/__init__.py
git commit -m "feat(history): export Interaction model and update protocol"
```

---

## Phase 2: MCP Tool Integration

### Task 2.1: Add Interaction Logging to generate_layout

**Files:**
- Modify: `src/mcp/tools/generate.py`

**Step 1: Write the failing test**

Add to `src/mcp/tools/test.py`:

```python
class TestInteractionLogging:
    """Tests for MCP tool interaction logging."""

    def test_generate_layout_logs_interaction(self, temp_history_manager):
        """Test that generate_layout logs an interaction."""
        from src.mcp.tools.generate import generate_layout

        # Generate a layout
        result = generate_layout(query="simple button")

        # Check interaction was logged
        manager = temp_history_manager
        interactions = manager.list_interactions(tool_name="generate_layout")

        assert len(interactions) >= 1
        interaction = interactions[0]
        assert interaction.tool_name == "generate_layout"
        assert interaction.request_params["query"] == "simple button"
        assert interaction.artifact_id == result.get("artifact_id")
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/tools/test.py::TestInteractionLogging::test_generate_layout_logs_interaction -v`
Expected: FAIL (interaction not logged)

**Step 3: Write implementation**

Modify `generate_layout` function in `src/mcp/tools/generate.py`. After the artifact is persisted (around line 170), add:

```python
# Persist to history if enabled
if persist:
    try:
        from src.history import GenerationStats, get_history_manager

        manager = get_history_manager()
        artifact = manager.store_artifact(
            query=query,
            layout=layout_dict,
            draft=draft,
            session_id=session_id,
            parent_id=parent_id,
            model=output.stats.final_model,
            temperature=temperature,
            provider=provider,
            stats=GenerationStats(
                attempts=output.stats.attempts,
                validation_retries=output.stats.validation_retries,
                json_repairs=output.stats.json_repairs,
                total_tokens=output.stats.total_tokens,
                final_model=output.stats.final_model,
            ),
            rag_example_ids=output.prompt_context.example_ids,
            tags=tags,
        )
        result["artifact_id"] = artifact.id
        logger.debug(f"Persisted artifact: {artifact.id}")

        # Log interaction
        manager.store_interaction(
            tool_name="generate_layout",
            request_params={
                "query": query,
                "model": model,
                "temperature": temperature,
                "provider": provider,
                "include_rag": include_rag,
                "session_id": session_id,
                "parent_id": parent_id,
                "tags": tags,
            },
            session_id=session_id,
            artifact_id=artifact.id,
            response_summary={
                "status": "success",
                "node_count": len(str(layout_dict)),
                "rag_used": rag_available,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to persist artifact: {e}")
        result["artifact_id"] = None
```

**Step 4: Run test to verify it passes**

Run: `pytest src/mcp/tools/test.py::TestInteractionLogging::test_generate_layout_logs_interaction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp/tools/generate.py src/mcp/tools/test.py
git commit -m "feat(mcp): log interactions for generate_layout"
```

---

### Task 2.2: Add Interaction Logging to refine_layout (with feedback)

**Files:**
- Modify: `src/mcp/tools/refine.py`

**Step 1: Write the failing test**

Add to `src/mcp/tools/test.py`:

```python
def test_refine_layout_logs_feedback(self, temp_history_manager):
    """Test that refine_layout logs feedback separately."""
    from src.mcp.tools.generate import generate_layout
    from src.mcp.tools.refine import refine_layout

    # First generate a layout
    original = generate_layout(query="login form")
    artifact_id = original["artifact_id"]

    # Refine with feedback
    feedback_text = "make the submit button larger"
    refined = refine_layout(artifact_id=artifact_id, feedback=feedback_text)

    # Check interaction was logged with feedback
    manager = temp_history_manager
    interactions = manager.list_interactions(tool_name="refine_layout")

    assert len(interactions) >= 1
    interaction = interactions[0]
    assert interaction.tool_name == "refine_layout"
    assert interaction.feedback == feedback_text
    assert interaction.request_params["artifact_id"] == artifact_id
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/tools/test.py::TestInteractionLogging::test_refine_layout_logs_feedback -v`
Expected: FAIL

**Step 3: Write implementation**

Modify `refine_layout` function in `src/mcp/tools/refine.py`. Replace the entire function:

```python
def refine_layout(
    artifact_id: str,
    feedback: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Refine an existing layout based on user feedback.

    Takes a previous layout and natural language feedback to generate
    an improved version. Creates a parent-child link for lineage tracking.

    Args:
        artifact_id: ID of the layout to refine.
        feedback: Natural language description of desired changes.
            Examples:
            - "move the sidebar to the right"
            - "add a search box to the header"
            - "make the buttons larger"
        model: LLM model to use (optional).
        temperature: Creativity level 0.0-2.0. Default: 0.7

    Returns:
        Dictionary with:
        - artifact_id: New artifact ID
        - layout: Refined layout JSON
        - draft: Updated text tree
        - stats: Generation statistics
        - parent_id: ID of the original layout

    Example:
        >>> original = generate_layout("login form")
        >>> refined = refine_layout(
        ...     original["artifact_id"],
        ...     "add forgot password link below the form"
        ... )
    """
    from src.history import get_history_manager

    from .generate import generate_layout

    # Get original artifact
    manager = get_history_manager()
    original = manager.get_artifact(artifact_id)

    if not original:
        raise ValueError(f"Artifact not found: {artifact_id}")

    # Build refinement query that includes original context
    refinement_query = (
        f"Refine this layout: {original.query}\n\n"
        f"Current structure:\n{original.draft}\n\n"
        f"User feedback: {feedback}\n\n"
        f"Generate an updated layout incorporating the feedback."
    )

    # Generate with parent link
    result = generate_layout(
        query=refinement_query,
        model=model,
        temperature=temperature,
        include_rag=True,
        persist=True,
        session_id=original.session_id,
        parent_id=artifact_id,
    )

    # Log refinement interaction with feedback stored separately
    manager.store_interaction(
        tool_name="refine_layout",
        request_params={
            "artifact_id": artifact_id,
            "model": model,
            "temperature": temperature,
        },
        session_id=original.session_id,
        artifact_id=result.get("artifact_id"),
        feedback=feedback,
        response_summary={
            "status": "success",
            "parent_id": artifact_id,
        },
    )

    result["parent_id"] = artifact_id
    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest src/mcp/tools/test.py::TestInteractionLogging::test_refine_layout_logs_feedback -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp/tools/refine.py
git commit -m "feat(mcp): log refine_layout interactions with feedback"
```

---

### Task 2.3: Add get_session_history MCP Tool

**Files:**
- Create: `src/mcp/tools/session_history.py`
- Modify: `src/mcp/server.py`

**Step 1: Write the failing test**

Add to `src/mcp/tools/test.py`:

```python
def test_get_session_history_tool(self, temp_history_manager):
    """Test the get_session_history MCP tool."""
    from src.mcp.tools.generate import generate_layout
    from src.mcp.tools.session_history import get_session_history

    # Generate some layouts
    generate_layout(query="login form")
    generate_layout(query="dashboard")

    # Get session history
    result = get_session_history()

    assert "timeline" in result
    assert "session_id" in result
    assert len(result["timeline"]) >= 2  # At least 2 interactions
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/tools/test.py::TestInteractionLogging::test_get_session_history_tool -v`
Expected: FAIL (module not found)

**Step 3: Create new tool module**

Create `src/mcp/tools/session_history.py`:

```python
"""Session history tools for MCP server.

Provides tools to reconstruct conversation context and audit agent behavior.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_session_history(
    session_id: str | None = None,
    limit: int = 50,
    include_artifacts: bool = True,
) -> dict[str, Any]:
    """Get session history for conversation reconstruction.

    Returns a chronological timeline of all tool calls and generated
    artifacts for a session. Use this to understand what has happened
    in a design conversation.

    Args:
        session_id: Session to get history for (uses current if None).
        limit: Maximum events to return. Default: 50
        include_artifacts: Include artifact details in timeline. Default: True

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - session_name: Session name if set
        - timeline: List of events in chronological order, each with:
            - type: "interaction" or "artifact"
            - timestamp: ISO timestamp
            - tool_name: For interactions, the tool called
            - query: For artifacts, the generation query
            - feedback: For refine_layout, the feedback given
        - artifact_count: Total artifacts in session
        - interaction_count: Total interactions logged

    Example:
        >>> history = get_session_history()
        >>> for event in history["timeline"]:
        ...     if event["type"] == "interaction":
        ...         print(f"{event['timestamp']}: {event['tool_name']}")
        ...     elif event.get("feedback"):
        ...         print(f"  Feedback: {event['feedback']}")
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    # Get or use current session
    if session_id is None:
        session = manager.get_or_create_session()
        session_id = session.id
    else:
        session = manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

    # Get timeline
    timeline = manager.get_session_timeline(
        session_id=session_id,
        include_artifacts=include_artifacts,
    )

    # Apply limit
    if limit and len(timeline) > limit:
        timeline = timeline[-limit:]  # Most recent

    # Get counts
    interactions = manager.list_interactions(session_id=session_id, limit=10000)
    artifacts = manager.list_artifacts(session_id=session_id, limit=10000)

    return {
        "session_id": session_id,
        "session_name": session.name,
        "timeline": timeline,
        "artifact_count": len(artifacts),
        "interaction_count": len(interactions),
    }


def get_artifact_history(
    artifact_id: str,
) -> dict[str, Any]:
    """Get the full history of an artifact including all refinements.

    Traces the lineage of an artifact and returns all interactions
    that led to its creation, including feedback given at each step.

    Args:
        artifact_id: Artifact to get history for.

    Returns:
        Dictionary containing:
        - artifact_id: The artifact ID
        - lineage: List of artifacts from root to this one
        - interactions: All interactions related to this lineage
        - feedback_chain: List of feedback strings in order

    Example:
        >>> history = get_artifact_history("abc-123")
        >>> for feedback in history["feedback_chain"]:
        ...     print(f"Refinement: {feedback}")
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    artifact = manager.get_artifact(artifact_id)
    if not artifact:
        raise ValueError(f"Artifact not found: {artifact_id}")

    # Get lineage
    lineage_data = manager.get_lineage(artifact_id)
    ancestors = lineage_data["ancestors"]

    # Build lineage chain (root to current)
    lineage_chain = list(reversed(ancestors)) + [artifact]

    # Get all interactions for artifacts in lineage
    all_interactions = []
    feedback_chain = []

    for art in lineage_chain:
        interactions = manager.list_interactions(artifact_id=art.id)
        for interaction in interactions:
            all_interactions.append({
                "artifact_id": art.id,
                "tool_name": interaction.tool_name,
                "timestamp": interaction.created_at.isoformat(),
                "feedback": interaction.feedback,
                "request_params": interaction.request_params,
            })
            if interaction.feedback:
                feedback_chain.append(interaction.feedback)

    return {
        "artifact_id": artifact_id,
        "lineage": [
            {
                "id": art.id,
                "query": art.query,
                "draft_preview": art.draft[:100] + "..." if len(art.draft) > 100 else art.draft,
                "parent_id": art.parent_id,
                "created_at": art.created_at.isoformat(),
            }
            for art in lineage_chain
        ],
        "interactions": all_interactions,
        "feedback_chain": feedback_chain,
    }


__all__ = [
    "get_session_history",
    "get_artifact_history",
]
```

**Step 4: Register tools in MCP server**

Add to `src/mcp/server.py` after the existing tools:

```python
@mcp.tool
def get_session_history(
    session_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get session history for conversation reconstruction.

    Returns a chronological timeline of all tool calls and generated
    artifacts. Use this to understand what has happened in a design
    conversation or to resume work after context loss.

    Args:
        session_id: Session to get history for (uses current if None).
        limit: Maximum events to return (1-200). Default: 50

    Returns:
        Dictionary with:
        - session_id: The session ID
        - timeline: Chronological list of events
        - artifact_count: Total artifacts
        - interaction_count: Total tool calls logged

    Example:
        >>> history = get_session_history()
        >>> for event in history["timeline"]:
        ...     print(f"{event['timestamp']}: {event.get('tool_name', event.get('query'))}")
    """
    limit = max(1, min(200, limit))

    from .tools.session_history import get_session_history as _get_session_history

    return _get_session_history(
        session_id=session_id,
        limit=limit,
        include_artifacts=True,
    )


@mcp.tool
def get_artifact_history(
    artifact_id: str,
) -> dict[str, Any]:
    """Get the full refinement history of an artifact.

    Traces all refinements that led to this artifact and returns
    the feedback given at each step. Essential for understanding
    how a design evolved.

    Args:
        artifact_id: Artifact to get history for.

    Returns:
        Dictionary with:
        - lineage: List of artifacts from root to current
        - feedback_chain: All feedback strings in order
        - interactions: Detailed interaction log

    Example:
        User: "How did we arrive at this design?"
        → get_artifact_history(artifact_id)
        → Shows: original query → feedback 1 → feedback 2 → current
    """
    from .tools.session_history import get_artifact_history as _get_artifact_history

    return _get_artifact_history(artifact_id=artifact_id)
```

**Step 5: Run test to verify it passes**

Run: `pytest src/mcp/tools/test.py::TestInteractionLogging::test_get_session_history_tool -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mcp/tools/session_history.py src/mcp/server.py
git commit -m "feat(mcp): add get_session_history and get_artifact_history tools"
```

---

## Phase 3: CLI History Commands

### Task 3.1: Add history list Command

**Files:**
- Modify: `__main__.py`

**Step 1: Define CLI structure**

Add to `__main__.py` (find the CLI command group section):

```python
@cli.group()
def history():
    """Manage generation history."""
    pass


@history.command("list")
@click.option("--sessions", "-s", is_flag=True, help="List sessions instead of artifacts")
@click.option("--limit", "-l", default=20, help="Number of items to show")
@click.option("--session-id", help="Filter by session ID")
def history_list(sessions: bool, limit: int, session_id: str | None):
    """List history artifacts or sessions.

    Examples:
        python . history list           # List recent artifacts
        python . history list -s        # List sessions
        python . history list -l 50     # Show 50 items
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    if sessions:
        items = manager.list_sessions(limit=limit)
        print(f"\n{'ID':<40} {'Name':<20} {'Artifacts':<10} {'Updated'}")
        print("-" * 85)
        for session in items:
            name = session.name or "(unnamed)"
            updated = session.updated_at.strftime("%Y-%m-%d %H:%M")
            print(f"{session.id:<40} {name:<20} {session.artifact_count:<10} {updated}")
    else:
        items = manager.list_artifacts(session_id=session_id, limit=limit)
        print(f"\n{'ID':<40} {'Query':<30} {'Created'}")
        print("-" * 85)
        for artifact in items:
            query = artifact.query[:27] + "..." if len(artifact.query) > 30 else artifact.query
            created = artifact.created_at.strftime("%Y-%m-%d %H:%M")
            print(f"{artifact.id:<40} {query:<30} {created}")

    print(f"\nTotal: {len(items)} items")
```

**Step 2: Run CLI to verify it works**

Run: `python . history list --help`
Expected: Shows help text

Run: `python . history list -s`
Expected: Shows sessions table

**Step 3: Commit**

```bash
git add __main__.py
git commit -m "feat(cli): add history list command"
```

---

### Task 3.2: Add history clear Command

**Files:**
- Modify: `__main__.py`

**Step 1: Add command**

Add to `__main__.py` after history_list:

```python
@history.command("clear")
@click.option("--session-id", help="Clear specific session only")
@click.option("--all", "-a", "clear_all", is_flag=True, help="Clear ALL history (destructive)")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def history_clear(session_id: str | None, clear_all: bool, force: bool):
    """Clear history data.

    Examples:
        python . history clear --session-id abc-123  # Clear one session
        python . history clear --all                 # Clear everything
        python . history clear --all -f              # No confirmation
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    if not session_id and not clear_all:
        print("Error: Specify --session-id or --all")
        raise SystemExit(1)

    if clear_all:
        stats = manager.get_stats()
        if not force:
            print(f"\nThis will delete:")
            print(f"  - {stats.artifact_count} artifacts")
            print(f"  - {stats.session_count} sessions")
            print(f"  - {stats.total_size_mb:.2f} MB of data")
            confirm = input("\nType 'yes' to confirm: ")
            if confirm.lower() != "yes":
                print("Cancelled.")
                return

        # Delete all sessions (CASCADE deletes artifacts)
        sessions = manager.list_sessions(limit=10000)
        for session in sessions:
            manager.delete_session(session.id)

        print(f"Cleared {stats.artifact_count} artifacts, {stats.session_count} sessions")

    elif session_id:
        session = manager.get_session(session_id)
        if not session:
            print(f"Session not found: {session_id}")
            raise SystemExit(1)

        if not force:
            print(f"\nSession: {session.name or session_id}")
            print(f"Artifacts: {session.artifact_count}")
            confirm = input("Delete? (yes/no): ")
            if confirm.lower() != "yes":
                print("Cancelled.")
                return

        manager.delete_session(session_id)
        print(f"Deleted session {session_id}")
```

**Step 2: Run CLI to verify**

Run: `python . history clear --help`
Expected: Shows help text

**Step 3: Commit**

```bash
git add __main__.py
git commit -m "feat(cli): add history clear command"
```

---

### Task 3.3: Add history stats Command

**Files:**
- Modify: `__main__.py`

**Step 1: Add command**

Add to `__main__.py`:

```python
@history.command("stats")
def history_stats():
    """Show history storage statistics.

    Example:
        python . history stats
    """
    from src.history import get_history_manager

    manager = get_history_manager()
    stats = manager.get_stats()

    print("\n=== History Storage Statistics ===")
    print(f"Total size:       {stats.total_size_mb:.2f} MB")
    print(f"Artifacts:        {stats.artifact_count}")
    print(f"Sessions:         {stats.session_count}")
    print(f"Variation sets:   {stats.variation_set_count}")
    print(f"Preview cache:    {stats.preview_cache_size_mb:.2f} MB")
    print(f"Oldest artifact:  {stats.oldest_artifact_age_days:.0f} days")
    print(f"Orphan count:     {stats.orphan_count}")

    # Show interactions count
    interactions = manager.list_interactions(limit=10000)
    print(f"Interactions:     {len(interactions)}")
```

**Step 2: Run CLI to verify**

Run: `python . history stats`
Expected: Shows statistics

**Step 3: Commit**

```bash
git add __main__.py
git commit -m "feat(cli): add history stats command"
```

---

### Task 3.4: Add history timeline Command

**Files:**
- Modify: `__main__.py`

**Step 1: Add command**

Add to `__main__.py`:

```python
@history.command("timeline")
@click.option("--session-id", help="Session to show timeline for")
@click.option("--limit", "-l", default=20, help="Number of events")
def history_timeline(session_id: str | None, limit: int):
    """Show session timeline for debugging.

    Example:
        python . history timeline
        python . history timeline --session-id abc-123
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    if session_id is None:
        session = manager.get_or_create_session()
        session_id = session.id

    timeline = manager.get_session_timeline(session_id=session_id)

    if not timeline:
        print("No events in timeline.")
        return

    # Show most recent
    timeline = timeline[-limit:]

    print(f"\n=== Session Timeline ({session_id[:8]}...) ===\n")

    for event in timeline:
        timestamp = event["timestamp"][:19]  # Truncate microseconds

        if event["type"] == "interaction":
            tool = event["tool_name"]
            feedback = event.get("feedback")
            if feedback:
                print(f"{timestamp}  [CALL] {tool}")
                print(f"                  Feedback: {feedback[:50]}...")
            else:
                print(f"{timestamp}  [CALL] {tool}")

        elif event["type"] == "artifact":
            query = event.get("query", "")[:40]
            artifact_id = event["artifact_id"][:8]
            print(f"{timestamp}  [GEN]  {artifact_id}... → {query}")

    print(f"\nTotal events: {len(timeline)}")
```

**Step 2: Run CLI to verify**

Run: `python . history timeline --help`
Expected: Shows help text

**Step 3: Commit**

```bash
git add __main__.py
git commit -m "feat(cli): add history timeline command"
```

---

## Phase 4: Update Reset Command to Include History Option

### Task 4.1: Add --history Flag to Reset

**Files:**
- Modify: `src/reset/lib.py`
- Modify: `__main__.py`

**Step 1: Add clear_history function**

Add to `src/reset/lib.py`:

```python
def clear_history(verbose: bool = True) -> bool:
    """Delete history database and preview cache.

    Removes:
        - data/history/history.db (SQLite database)
        - data/history/previews/ (preview cache)

    Args:
        verbose: Print progress messages.

    Returns:
        True if all deletions succeeded.
    """
    _print("[reset] Clearing history database...", verbose)
    success = True

    try:
        repo_root = find_repo_root()
        history_dir = repo_root / "data" / "history"

        if history_dir.exists():
            # Close any open connections first
            from src.history import close_history_manager
            close_history_manager()

            for item in history_dir.iterdir():
                if not _delete_path(item, verbose):
                    success = False
        else:
            _print("  History directory does not exist", verbose)

    except Exception as e:
        _print(f"  Failed to clear history: {e}", verbose)
        success = False

    return success
```

Update `reset_environment` function:

```python
def reset_environment(
    index: bool = False,
    models: bool = False,
    docker: bool = False,
    temp: bool = False,
    history: bool = False,
    verbose: bool = True,
) -> bool:
    """Reset development environment artifacts.

    Orchestrates cleanup based on flags. If no flags are set,
    defaults to index-only reset.

    Args:
        index: Clear RAG indices.
        models: Clear embedding models.
        docker: Purge Docker stack.
        temp: Clear temporary files.
        history: Clear history database.
        verbose: Print progress messages.

    Returns:
        True if all requested operations succeeded.
    """
    # Default to index if nothing specified
    if not any([index, models, docker, temp, history]):
        index = True

    success = True

    if index:
        if not clear_indices(verbose):
            success = False

    if models:
        if not clear_models(verbose):
            success = False

    if docker:
        if not clear_docker(verbose):
            success = False

    if temp:
        if not clear_temp(verbose):
            success = False

    if history:
        if not clear_history(verbose):
            success = False

    if success:
        _print("\n[reset] Done. Environment reset complete.", verbose)
    else:
        _print("\n[reset] Completed with errors.", verbose)

    return success
```

Update `__all__` in `src/reset/lib.py`:

```python
__all__ = [
    "clear_indices",
    "clear_models",
    "clear_docker",
    "clear_temp",
    "clear_history",
    "reset_environment",
]
```

**Step 2: Update CLI command**

Modify the reset command in `__main__.py`:

```python
@click.option("--history", "-H", is_flag=True, help="Clear history database")
```

And update the function call:

```python
reset_environment(
    index=index,
    models=models,
    docker=docker,
    temp=temp,
    history=history,
    verbose=True,
)
```

**Step 3: Run CLI to verify**

Run: `python . reset --help`
Expected: Shows --history flag

**Step 4: Commit**

```bash
git add src/reset/lib.py __main__.py
git commit -m "feat(reset): add --history flag to clear history database"
```

---

## Phase 5: Final Integration Tests

### Task 5.1: Add Integration Test for Full Workflow

**Files:**
- Modify: `src/history/test.py`

**Step 1: Write integration test**

Add to `src/history/test.py`:

```python
class TestFullWorkflow:
    """Integration tests for complete history workflow."""

    def test_generate_refine_timeline(self, manager, sample_layout, sample_stats):
        """Test full workflow: generate → refine → timeline."""
        from src.history.models import Interaction

        session = manager.get_or_create_session("workflow-test")

        # Simulate generate_layout
        artifact1 = manager.store_artifact(
            query="login form",
            layout=sample_layout,
            draft="Login Form [container]",
            session_id=session.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        interaction1 = Interaction.create(
            session_id=session.id,
            tool_name="generate_layout",
            request_params={"query": "login form"},
            artifact_id=artifact1.id,
        )
        manager.store_interaction(interaction1)

        # Simulate refine_layout
        artifact2 = manager.store_artifact(
            query="Refine: login form with forgot password",
            layout=sample_layout,
            draft="Login Form [container]\n└── Forgot Password [link]",
            session_id=session.id,
            parent_id=artifact1.id,
            model="gpt-4.1-mini",
            temperature=0.7,
            stats=sample_stats,
        )

        interaction2 = Interaction.create(
            session_id=session.id,
            tool_name="refine_layout",
            request_params={"artifact_id": artifact1.id},
            artifact_id=artifact2.id,
            feedback="add forgot password link",
        )
        manager.store_interaction(interaction2)

        # Verify timeline
        timeline = manager.get_session_timeline(session_id=session.id)
        assert len(timeline) == 4  # 2 interactions + 2 artifacts

        # Verify feedback is preserved
        refine_events = [e for e in timeline if e.get("feedback")]
        assert len(refine_events) == 1
        assert refine_events[0]["feedback"] == "add forgot password link"

        # Verify lineage
        lineage = manager.get_lineage(artifact2.id)
        assert artifact1.id in [a.id for a in lineage["ancestors"]]
```

**Step 2: Run test**

Run: `pytest src/history/test.py::TestFullWorkflow -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/history/test.py
git commit -m "test(history): add full workflow integration test"
```

---

### Task 5.2: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest src/history/test.py -v`
Expected: All tests PASS

Run: `pytest src/mcp/tools/test.py -v`
Expected: All tests PASS

**Step 2: Run linter**

Run: `ruff check src/history src/mcp/tools src/reset`
Expected: No errors

Run: `ruff format src/history src/mcp/tools src/reset`
Expected: Files formatted

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: format and lint interaction logging implementation"
```

---

## Summary

This plan implements:

1. **Interaction Model** - New `Interaction` dataclass for logging MCP tool calls
2. **Database Schema** - New `interactions` table with proper indexes and foreign keys
3. **Storage Layer** - Methods in `SQLiteStorage` and `HistoryManager`
4. **MCP Tool Logging** - `generate_layout` and `refine_layout` log interactions
5. **New MCP Tools** - `get_session_history` and `get_artifact_history`
6. **CLI Commands** - `history list`, `history clear`, `history stats`, `history timeline`
7. **Reset Integration** - `--history` flag for clearing history

**No deprecated code** - all changes are additive or complete replacements.

**Total commits:** ~12 focused commits with clear messages.
