# History Manager API Design

> Orchestrating variation generation, archival, and retrieval for wireframe-mcp

---

## Problem Statement

Currently, the MCP server is **stateless**:
- Generated layouts exist only as return values
- No way to request "give me 3 variations"
- No history of past generations
- No linking between related layouts (original → refinements)
- No cross-referencing MCP calls to artifacts

This makes it impossible to:
1. Compare variations of a layout
2. Retrieve past work
3. Track layout evolution
4. Build on previous generations

---

## Design Goals

1. **Artifact Persistence**: Store all generated layouts with full metadata
2. **Session Management**: Group related operations under sessions
3. **Variation Support**: Generate N variations and rank/compare them
4. **Lineage Tracking**: Link originals → refinements → variations
5. **Cross-Reference**: Track which RAG examples influenced which layouts
6. **Efficient Retrieval**: Query by session, time, query similarity, or metadata

---

## Core Data Model

### 1. GenerationArtifact

The primary stored entity for any generated layout.

```python
@dataclass
class GenerationArtifact:
    """A persisted generated layout with full metadata."""

    # Identity
    id: str                          # UUID for this artifact
    session_id: str                  # Parent session
    parent_id: str | None            # For refinements/variations

    # Content
    layout: LayoutNode               # The actual layout structure
    draft: str                       # Text tree representation
    query: str                       # Original natural language query

    # Generation Context
    model: str                       # LLM model used
    temperature: float               # Generation temperature
    provider: str                    # DSL provider hint

    # Statistics
    stats: GenerationStats           # Attempts, tokens, retries

    # RAG Lineage
    rag_example_ids: list[str]       # Which corpus items were used
    rag_scores: dict[str, float]     # Relevance scores for each

    # Grouping
    variation_group: str | None      # Groups variations together
    variation_index: int | None      # Position in variation set
    tags: list[str]                  # User-defined tags

    # Timestamps
    created_at: datetime

    # Rendered Previews (optional, cached)
    previews: dict[str, bytes] | None  # style -> image bytes
```

### 2. Session

Groups related generation operations.

```python
@dataclass
class Session:
    """A logical grouping of related generation operations."""

    id: str                          # UUID
    name: str | None                 # Optional user-friendly name
    created_at: datetime
    updated_at: datetime

    # Metadata
    description: str | None
    tags: list[str]

    # Statistics
    artifact_count: int
    total_tokens: int
```

### 3. VariationRequest

Request structure for generating multiple variations.

```python
@dataclass
class VariationRequest:
    """Request to generate N variations of a layout."""

    query: str                       # Base query
    count: int                       # Number of variations (1-10)

    # Optional constraints
    base_artifact_id: str | None     # Generate variations of existing
    temperature_range: tuple[float, float] | None  # e.g., (0.5, 1.0)
    style_variety: bool              # Vary visual styles too

    # Generation options
    model: str | None
    include_rag: bool = True
```

### 4. VariationSet

Result of a variation request.

```python
@dataclass
class VariationSet:
    """A set of related layout variations."""

    id: str                          # UUID for this set
    session_id: str
    request: VariationRequest

    # Results
    artifacts: list[GenerationArtifact]

    # Comparison metrics
    diversity_score: float           # How different are variations
    rankings: dict[str, list[str]]   # criterion -> ordered artifact IDs

    created_at: datetime
```

---

## History Manager Interface

```python
class HistoryManager(Protocol):
    """Interface for managing generation history and variations."""

    # ==========================================================================
    # Session Management
    # ==========================================================================

    def create_session(
        self,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> Session:
        """Create a new session for grouping related operations."""
        ...

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        ...

    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        tags: list[str] | None = None,
    ) -> list[Session]:
        """List sessions with optional filtering."""
        ...

    def get_current_session(self) -> Session:
        """Get or create the current working session."""
        ...

    # ==========================================================================
    # Artifact Storage
    # ==========================================================================

    def store_artifact(
        self,
        layout: LayoutNode,
        query: str,
        stats: GenerationStats,
        rag_context: PromptContext,
        session_id: str | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> GenerationArtifact:
        """Store a generated layout as an artifact."""
        ...

    def get_artifact(self, artifact_id: str) -> GenerationArtifact | None:
        """Retrieve an artifact by ID."""
        ...

    def list_artifacts(
        self,
        session_id: str | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GenerationArtifact]:
        """List artifacts with filtering options."""
        ...

    def search_artifacts(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
    ) -> list[tuple[GenerationArtifact, float]]:
        """Semantic search over stored artifacts."""
        ...

    # ==========================================================================
    # Variation Generation
    # ==========================================================================

    def generate_variations(
        self,
        request: VariationRequest,
        session_id: str | None = None,
    ) -> VariationSet:
        """Generate N variations of a layout query."""
        ...

    def get_variation_set(self, set_id: str) -> VariationSet | None:
        """Retrieve a variation set by ID."""
        ...

    def compare_artifacts(
        self,
        artifact_ids: list[str],
        criteria: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple artifacts on various criteria."""
        ...

    # ==========================================================================
    # Lineage & Cross-Reference
    # ==========================================================================

    def get_lineage(
        self,
        artifact_id: str,
        direction: Literal["ancestors", "descendants", "both"] = "both",
    ) -> dict[str, list[GenerationArtifact]]:
        """Get parent/child relationships for an artifact."""
        ...

    def get_rag_influence(
        self,
        artifact_id: str,
    ) -> list[tuple[str, float, str]]:
        """Get RAG examples that influenced a generation.

        Returns: List of (example_id, score, serialized_text)
        """
        ...

    def find_artifacts_using_example(
        self,
        example_id: str,
    ) -> list[GenerationArtifact]:
        """Find all artifacts influenced by a specific corpus example."""
        ...

    # ==========================================================================
    # Preview Caching
    # ==========================================================================

    def cache_preview(
        self,
        artifact_id: str,
        style: str,
        image_bytes: bytes,
        format: str = "png",
    ) -> None:
        """Cache a rendered preview for an artifact."""
        ...

    def get_cached_preview(
        self,
        artifact_id: str,
        style: str,
    ) -> bytes | None:
        """Retrieve a cached preview if available."""
        ...
```

---

## Storage Backend Options

### Option A: SQLite + FAISS (Recommended for Local)

```
data/history/
├── history.db           # SQLite: sessions, artifacts, lineage
├── artifacts.faiss      # FAISS: artifact embedding index
└── previews/            # File-based preview cache
    ├── {artifact_id}/
    │   ├── wireframe.png
    │   ├── sketch.png
    │   └── minimal.png
```

**Pros**: Simple, no external deps, portable
**Cons**: Single-user, no concurrent writes

### Option B: PostgreSQL + pgvector

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    name TEXT,
    description TEXT,
    tags TEXT[],
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);

CREATE TABLE artifacts (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    parent_id UUID REFERENCES artifacts(id),
    layout JSONB NOT NULL,
    draft TEXT NOT NULL,
    query TEXT NOT NULL,
    model TEXT,
    temperature FLOAT,
    stats JSONB,
    rag_example_ids TEXT[],
    variation_group UUID,
    variation_index INT,
    tags TEXT[],
    embedding vector(1536),
    created_at TIMESTAMPTZ
);

CREATE INDEX ON artifacts USING ivfflat (embedding vector_cosine_ops);
```

**Pros**: Production-ready, concurrent, vector search built-in
**Cons**: Requires PostgreSQL setup

### Option C: In-Memory with File Backup

For development/testing - keeps all in memory, periodic JSON dump.

---

## MCP Tool Integration

### New Tools

```python
@mcp.tool
def generate_variations(
    query: str,
    count: int = 3,
    temperature_range: tuple[float, float] | None = None,
    base_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Generate multiple layout variations for comparison.

    Args:
        query: Natural language description
        count: Number of variations (1-10)
        temperature_range: Optional (min, max) temperature spread
        base_artifact_id: Generate variations of existing artifact

    Returns:
        Dictionary with:
        - variation_set_id: ID for this set
        - artifacts: List of generated artifacts with IDs
        - comparison: Basic comparison metrics
    """
    ...

@mcp.tool
def get_history(
    session_id: str | None = None,
    limit: int = 20,
    query: str | None = None,
) -> dict[str, Any]:
    """Retrieve generation history.

    Args:
        session_id: Filter by session
        limit: Max results
        query: Semantic search within history

    Returns:
        Dictionary with:
        - artifacts: List of artifact summaries
        - total_count: Total matching artifacts
    """
    ...

@mcp.tool
def get_artifact(
    artifact_id: str,
    include_preview: bool = False,
) -> dict[str, Any]:
    """Retrieve a specific artifact by ID.

    Args:
        artifact_id: The artifact UUID
        include_preview: Include cached preview images

    Returns:
        Full artifact data including layout, stats, lineage
    """
    ...

@mcp.tool
def refine_layout(
    artifact_id: str,
    refinement: str,
    preserve_structure: bool = True,
) -> dict[str, Any]:
    """Refine an existing layout with modifications.

    Args:
        artifact_id: Base artifact to refine
        refinement: Natural language modification request
        preserve_structure: Keep overall structure, modify details

    Returns:
        New artifact linked to parent
    """
    ...

@mcp.tool
def compare_layouts(
    artifact_ids: list[str],
) -> dict[str, Any]:
    """Compare multiple layouts.

    Args:
        artifact_ids: List of artifact IDs to compare

    Returns:
        Comparison metrics: structure similarity, component overlap, etc.
    """
    ...
```

### Modified `generate_layout`

```python
@mcp.tool
def generate_layout(
    query: str,
    model: str | None = None,
    temperature: float = 0.7,
    provider: str = "d2",
    include_rag: bool = True,
    # NEW PARAMETERS
    session_id: str | None = None,   # Attach to session
    parent_id: str | None = None,    # Link to parent artifact
    tags: list[str] | None = None,   # User tags
    persist: bool = True,            # Store in history (default True)
) -> dict[str, Any]:
    """Generate a UI layout from natural language description.

    Returns:
        Dictionary with:
        - artifact_id: UUID of stored artifact (if persist=True)
        - layout: Generated LayoutNode as JSON
        - draft: Human-readable text tree
        - stats: Generation statistics
        - rag_examples: IDs of corpus items used
    """
    ...
```

---

## Variation Generation Algorithm

```python
async def generate_variations(
    request: VariationRequest,
    generator: LayoutGenerator,
    history: HistoryManager,
) -> VariationSet:
    """Generate N variations with diversity optimization."""

    # 1. Determine temperature spread
    if request.temperature_range:
        temps = np.linspace(
            request.temperature_range[0],
            request.temperature_range[1],
            request.count
        )
    else:
        # Default: spread around 0.7
        temps = np.linspace(0.5, 1.0, request.count)

    # 2. Generate variations concurrently
    artifacts = []
    variation_group = str(uuid4())

    async def generate_one(temp: float, index: int) -> GenerationArtifact:
        output = await generator.generate_async(
            query=request.query,
            temperature=temp,
        )
        return history.store_artifact(
            layout=output.context.node,
            query=request.query,
            stats=output.stats,
            rag_context=output.prompt_context,
            variation_group=variation_group,
            variation_index=index,
        )

    artifacts = await asyncio.gather(*[
        generate_one(temp, i)
        for i, temp in enumerate(temps)
    ])

    # 3. Calculate diversity metrics
    diversity = calculate_structural_diversity(artifacts)

    # 4. Rank by various criteria
    rankings = {
        "complexity": rank_by_complexity(artifacts),
        "depth": rank_by_depth(artifacts),
        "component_variety": rank_by_component_variety(artifacts),
    }

    return VariationSet(
        id=str(uuid4()),
        session_id=history.get_current_session().id,
        request=request,
        artifacts=artifacts,
        diversity_score=diversity,
        rankings=rankings,
        created_at=datetime.now(UTC),
    )
```

---

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Data Models** (`src/history/models.py`)
   - `GenerationArtifact`, `Session`, `VariationSet` dataclasses
   - Pydantic models for serialization

2. **Storage Backend** (`src/history/storage/`)
   - `HistoryStorage` protocol
   - `SQLiteStorage` implementation
   - `InMemoryStorage` for testing

3. **History Manager** (`src/history/lib.py`)
   - Core `HistoryManager` implementation
   - Session management
   - Artifact CRUD

### Phase 2: Integration

4. **Generator Integration** (`src/llm/generator/lib.py`)
   - Modify `LayoutGenerator` to optionally persist
   - Add artifact linking to generation flow

5. **MCP Tool Updates** (`src/mcp/tools/`)
   - Update `generate_layout` with persistence params
   - Add `get_history`, `get_artifact` tools

### Phase 3: Variations

6. **Variation Engine** (`src/history/variations.py`)
   - `generate_variations()` implementation
   - Diversity metrics
   - Ranking algorithms

7. **New MCP Tools**
   - `generate_variations` tool
   - `compare_layouts` tool
   - `refine_layout` tool

### Phase 4: Lineage & Search

8. **Lineage Tracking** (`src/history/lineage.py`)
   - Parent-child relationships
   - RAG influence tracking

9. **Semantic Search**
   - Artifact embedding index
   - `search_artifacts` implementation

---

## Configuration

```python
@dataclass
class HistoryConfig:
    """Configuration for history management."""

    # Storage
    backend: Literal["sqlite", "postgres", "memory"] = "sqlite"
    db_path: Path = Path("data/history/history.db")

    # Behavior
    auto_persist: bool = True        # Auto-save all generations
    auto_preview: bool = False       # Auto-generate previews

    # Retention
    max_artifacts_per_session: int = 1000
    artifact_retention_days: int = 90

    # Variations
    max_variations_per_request: int = 10
    default_variation_count: int = 3

    # Preview cache
    cache_previews: bool = True
    preview_cache_dir: Path = Path("data/history/previews")
```

---

## Example Usage Flow

```
User: "Generate 3 variations of a settings page with toggles"

→ Agent calls: generate_variations(
    query="settings page with toggles",
    count=3
  )

→ System:
  1. Creates variation_group UUID
  2. Generates 3 layouts at temps [0.5, 0.75, 1.0]
  3. Stores all 3 as linked artifacts
  4. Calculates diversity metrics
  5. Returns variation set with comparisons

← Returns:
  {
    "variation_set_id": "abc123",
    "artifacts": [
      {"id": "v1", "draft": "...", "temperature": 0.5},
      {"id": "v2", "draft": "...", "temperature": 0.75},
      {"id": "v3", "draft": "...", "temperature": 1.0},
    ],
    "diversity_score": 0.73,
    "rankings": {
      "complexity": ["v3", "v2", "v1"],
      "depth": ["v2", "v3", "v1"]
    }
  }

User: "I like variation 2, but add a header section"

→ Agent calls: refine_layout(
    artifact_id="v2",
    refinement="add a header section at the top"
  )

→ System:
  1. Loads artifact v2
  2. Generates refinement with v2 as context
  3. Stores new artifact with parent_id="v2"
  4. Returns new artifact

← Returns:
  {
    "artifact_id": "v2-refined",
    "parent_id": "v2",
    "layout": {...},
    "draft": "Settings [container]\n├── Header [toolbar]..."
  }
```

---

## File Structure

```
src/history/
├── __init__.py
├── lib.py              # HistoryManager implementation
├── models.py           # Data models (Artifact, Session, etc.)
├── variations.py       # Variation generation engine
├── lineage.py          # Lineage tracking utilities
├── storage/
│   ├── __init__.py
│   ├── protocol.py     # HistoryStorage protocol
│   ├── sqlite.py       # SQLite implementation
│   ├── memory.py       # In-memory implementation
│   └── postgres.py     # PostgreSQL implementation (optional)
└── test.py             # Unit tests
```

---

## Success Criteria

1. **Persistence**: All generated layouts retrievable by ID
2. **Variations**: Can generate N variations with single request
3. **History**: Can browse/search past generations
4. **Lineage**: Can trace refinement chains
5. **Cross-Reference**: Can see which RAG examples influenced generations
6. **Performance**: <100ms for artifact retrieval, <5s for variation sets
