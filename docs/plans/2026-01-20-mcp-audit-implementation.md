# MCP Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs, complete the MCP workflow, add quality measurement, and validate the RAG value proposition.

**Architecture:** TDD approach - write failing tests first, then minimal implementation. Each phase delivers standalone value.

**Tech Stack:** Python 3.12, pytest, FastMCP 2.x, FAISS, Pydantic 2.x

---

## Phase 1: Critical Bugs (3 Tasks)

### Task 1.1: Fix prompt_context.examples Crash Bug

**Files:**
- Modify: `src/mcp/tools/generate.py:128-129`
- Test: `src/mcp/tools/test.py` (create if needed)

**Context:** Lines 128-129 reference `output.prompt_context.examples` but PromptContext only has `example_count` and `example_ids` fields.

**Step 1: Write the failing test**

Create `src/mcp/tools/test.py`:

```python
"""Unit tests for MCP tools."""

import pytest


class TestGenerateLayoutStats:
    """Tests for generate_layout stats output."""

    @pytest.mark.unit
    def test_stats_rag_fields_exist(self, mock_generator_output):
        """Stats includes correct RAG field names."""
        from src.mcp.tools.generate import generate_layout

        # This should NOT crash - currently crashes on prompt_context.examples
        result = generate_layout(
            query="simple login form",
            include_rag=False,  # Simpler path
            persist=False,
        )

        assert "stats" in result
        assert "rag_used" in result["stats"]
        assert "rag_examples" in result["stats"]
        assert isinstance(result["stats"]["rag_used"], bool)
        assert isinstance(result["stats"]["rag_examples"], int)


@pytest.fixture
def mock_generator_output(monkeypatch):
    """Mock LayoutGenerator to avoid LLM calls."""
    from unittest.mock import MagicMock

    from src.mid import LayoutNode
    from src.schema import ComponentType

    mock_node = LayoutNode(id="root", type=ComponentType.CONTAINER)

    class MockContext:
        node = mock_node

    class MockPromptContext:
        example_count = 0
        example_ids = []

    class MockStats:
        attempts = 1
        validation_retries = 0
        json_repairs = 0
        total_tokens = 100
        final_model = "mock-model"

    class MockOutput:
        context = MockContext()
        prompt_context = MockPromptContext()
        stats = MockStats()

    def mock_generate(*args, **kwargs):
        return MockOutput()

    monkeypatch.setattr(
        "src.llm.generator.lib.LayoutGenerator.generate", mock_generate
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/tools/test.py::TestGenerateLayoutStats::test_stats_rag_fields_exist -v`
Expected: FAIL with `AttributeError: 'MockPromptContext' object has no attribute 'examples'`

**Step 3: Fix the bug in generate.py**

Edit `src/mcp/tools/generate.py` lines 128-129:

```python
# BEFORE (buggy):
"rag_used": rag_available and len(output.prompt_context.examples) > 0,
"rag_examples": len(output.prompt_context.examples),

# AFTER (fixed):
"rag_used": rag_available and output.prompt_context.example_count > 0,
"rag_examples": output.prompt_context.example_count,
```

**Step 4: Run test to verify it passes**

Run: `pytest src/mcp/tools/test.py::TestGenerateLayoutStats::test_stats_rag_fields_exist -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp/tools/generate.py src/mcp/tools/test.py
git commit -m "fix(mcp): use correct prompt_context fields for RAG stats"
```

---

### Task 1.2: Register get_artifact Tool

**Files:**
- Modify: `src/mcp/server.py` (add tool registration)
- Modify: `src/mcp/test.py` (update tool count assertion)

**Context:** `get_artifact` is required to retrieve full layout from variations workflow but is not exposed via MCP.

**Step 1: Write the failing test**

Add to `src/mcp/test.py` in `TestToolRegistration`:

```python
@pytest.mark.unit
def test_get_artifact_tool_registered(self):
    """get_artifact tool is registered for variations workflow."""
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "get_artifact" in tool_names
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/test.py::TestToolRegistration::test_get_artifact_tool_registered -v`
Expected: FAIL with `AssertionError: assert 'get_artifact' in {...}`

**Step 3: Register the tool in server.py**

Add after line 267 (after `generate_variations`):

```python
@mcp.tool
def get_artifact(
    artifact_id: str,
    include_lineage: bool = False,
) -> dict[str, Any]:
    """Retrieve a previously generated layout by its artifact ID.

    Use this after generate_variations to get the full layout for
    a specific variation you want to preview.

    Args:
        artifact_id: The artifact UUID from generate_layout or variations.
        include_lineage: Include parent/child relationships. Default: False

    Returns:
        Dictionary with:
        - artifact: Full artifact data including layout, draft, stats
        - lineage: Parent/child IDs (if include_lineage=True)

    Example workflow:
        1. generate_variations("dashboard", count=3) → get artifact IDs
        2. User picks variation 2
        3. get_artifact(artifact_id) → get full layout
        4. preview_layout(layout) → render wireframe
    """
    from .tools.history import get_artifact as _get_artifact

    return _get_artifact(artifact_id=artifact_id, include_lineage=include_lineage)
```

**Step 4: Update tool count in test**

Edit `src/mcp/test.py` line 156:

```python
# BEFORE:
assert len(tool_names) == 6

# AFTER:
assert len(tool_names) == 7
```

Also remove line 167 (the assertion that get_artifact should NOT be exposed):

```python
# DELETE this line:
assert "get_artifact" not in tool_names
```

**Step 5: Run tests to verify they pass**

Run: `pytest src/mcp/test.py::TestToolRegistration -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/mcp/server.py src/mcp/test.py
git commit -m "feat(mcp): register get_artifact tool for variations workflow"
```

---

### Task 1.3: Add MCP Conftest for FastMCP Fixture

**Files:**
- Create: `src/mcp/conftest.py`
- Modify: `requirements.txt` (ensure fastmcp in dev deps)

**Context:** MCP protocol tests skip with `pytest.skip("fastmcp not installed")` - need proper fixture setup.

**Step 1: Write the failing test**

Add to `src/mcp/test.py` at the end of `TestMCPProtocol`:

```python
@pytest.mark.asyncio
async def test_client_can_call_get_artifact_not_found(self, mcp_client):
    """get_artifact returns error for unknown ID."""
    result = await mcp_client.call_tool(
        "get_artifact",
        {"artifact_id": "nonexistent-id-12345"}
    )

    # Should return error, not crash
    assert result is not None
```

**Step 2: Run test to verify setup fails**

Run: `pytest src/mcp/test.py::TestMCPProtocol::test_client_can_call_get_artifact_not_found -v`
Expected: SKIP with "fastmcp not installed" or fixture error

**Step 3: Create conftest.py with fixtures**

Create `src/mcp/conftest.py`:

```python
"""Pytest fixtures for MCP server tests."""

import pytest

# Check if fastmcp is available
try:
    from fastmcp import Client

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "mcp: MCP protocol integration tests")


def pytest_collection_modifyitems(config, items):
    """Skip MCP tests if fastmcp not installed."""
    if not FASTMCP_AVAILABLE:
        skip_mcp = pytest.mark.skip(reason="fastmcp not installed")
        for item in items:
            if "mcp" in item.keywords:
                item.add_marker(skip_mcp)


@pytest.fixture
def mcp_server():
    """Create MCP server instance for testing."""
    if not FASTMCP_AVAILABLE:
        pytest.skip("fastmcp not installed")

    from .server import create_server

    return create_server()


@pytest.fixture
async def mcp_client(mcp_server):
    """Create connected MCP client for testing."""
    if not FASTMCP_AVAILABLE:
        pytest.skip("fastmcp not installed")

    from fastmcp import Client

    async with Client(mcp_server) as client:
        yield client
```

**Step 4: Update TestMCPProtocol to use fixture**

Edit `src/mcp/test.py` TestMCPProtocol class to use the fixture instead of inline try/except:

```python
@pytest.mark.mcp
class TestMCPProtocol:
    """Integration tests using MCP client protocol."""

    @pytest.mark.asyncio
    async def test_client_can_connect(self, mcp_client):
        """Client can connect to server."""
        assert mcp_client is not None

    @pytest.mark.asyncio
    async def test_client_can_list_tools(self, mcp_client):
        """Client can list available tools."""
        tools = await mcp_client.list_tools()

        tool_names = [t.name for t in tools]
        assert "status" in tool_names
        assert "generate_layout" in tool_names
        assert "get_artifact" in tool_names

    @pytest.mark.asyncio
    async def test_client_can_call_status(self, mcp_client):
        """Client can call status tool."""
        result = await mcp_client.call_tool("status", {})
        assert result is not None

    @pytest.mark.asyncio
    async def test_client_can_call_get_artifact_not_found(self, mcp_client):
        """get_artifact returns error for unknown ID."""
        # Tool should handle error gracefully
        with pytest.raises(Exception, match="not found"):
            await mcp_client.call_tool(
                "get_artifact",
                {"artifact_id": "nonexistent-id-12345"}
            )
```

**Step 5: Run tests**

Run: `pytest src/mcp/test.py::TestMCPProtocol -v`
Expected: SKIP (if fastmcp not installed) or PASS (if installed)

**Step 6: Commit**

```bash
git add src/mcp/conftest.py src/mcp/test.py
git commit -m "test(mcp): add conftest with proper fastmcp fixtures"
```

---

## Phase 2: Workflow Completion (4 Tasks)

### Task 2.1: Register validate_layout Tool

**Files:**
- Modify: `src/mcp/server.py`
- Modify: `src/mcp/test.py`

**Step 1: Write the failing test**

Add to `src/mcp/test.py` TestToolRegistration:

```python
@pytest.mark.unit
def test_validate_layout_tool_registered(self):
    """validate_layout tool is registered for pre-implementation checks."""
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "validate_layout" in tool_names
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/test.py::TestToolRegistration::test_validate_layout_tool_registered -v`
Expected: FAIL

**Step 3: Register tool in server.py**

Add after `get_artifact`:

```python
@mcp.tool
def validate_layout(
    layout: dict[str, Any],
) -> dict[str, Any]:
    """Validate a layout structure before implementation.

    Use this before handing off to a coding agent to catch issues early.
    Checks structure, constraints, and accessibility.

    Args:
        layout: Layout JSON from generate_layout.

    Returns:
        Dictionary with:
        - valid: True if layout passes all checks
        - errors: List of critical issues that must be fixed
        - warnings: List of non-critical suggestions
        - stats: Layout statistics (node count, depth, etc.)

    Example:
        >>> result = validate_layout(layout)
        >>> if not result["valid"]:
        ...     print("Fix these:", result["errors"])
    """
    from .tools.validate import validate_layout as _validate

    return _validate(layout=layout)
```

**Step 4: Update test count**

Edit `src/mcp/test.py`:
- Update tool count to 8
- Remove assertion that validate_layout should NOT be exposed

**Step 5: Run tests**

Run: `pytest src/mcp/test.py::TestToolRegistration -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mcp/server.py src/mcp/test.py
git commit -m "feat(mcp): register validate_layout for pre-implementation checks"
```

---

### Task 2.2: Register search_layouts Tool

**Files:**
- Modify: `src/mcp/server.py`
- Modify: `src/mcp/test.py`

**Step 1: Write the failing test**

```python
@pytest.mark.unit
def test_search_layouts_tool_registered(self):
    """search_layouts tool is registered for corpus exploration."""
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "search_layouts" in tool_names
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/test.py::TestToolRegistration::test_search_layouts_tool_registered -v`
Expected: FAIL

**Step 3: Register tool in server.py**

```python
@mcp.tool
def search_layouts(
    query: str,
    k: int = 5,
) -> dict[str, Any]:
    """Search for similar layouts in the corpus.

    Use this to find inspiration or see what layouts exist
    for similar UI patterns.

    Args:
        query: Natural language description to search for.
        k: Number of results (1-20). Default: 5

    Returns:
        Dictionary with:
        - results: List of similar layouts with scores
        - total_in_index: Total layouts in the index

    Note: Requires RAG index to be built.
    """
    if not 1 <= k <= 20:
        raise ValueError(f"k must be 1-20, got {k}")

    from .tools.search import search_layouts as _search

    return _search(query=query, k=k)
```

**Step 4: Update test count to 9, remove exclusion assertion**

**Step 5: Run tests and commit**

```bash
git add src/mcp/server.py src/mcp/test.py
git commit -m "feat(mcp): register search_layouts for corpus exploration"
```

---

### Task 2.3: Add refine_layout Tool

**Files:**
- Create: `src/mcp/tools/refine.py`
- Modify: `src/mcp/server.py`
- Modify: `src/mcp/tools/__init__.py`

**Context:** Users need to iterate on layouts with feedback, not regenerate from scratch.

**Step 1: Write the failing test**

Add to `src/mcp/tools/test.py`:

```python
class TestRefineLayout:
    """Tests for refine_layout tool."""

    @pytest.mark.unit
    def test_refine_layout_exists(self):
        """refine_layout function is importable."""
        from src.mcp.tools.refine import refine_layout

        assert callable(refine_layout)

    @pytest.mark.unit
    def test_refine_layout_requires_artifact_id(self):
        """refine_layout requires artifact_id parameter."""
        from src.mcp.tools.refine import refine_layout
        import inspect

        sig = inspect.signature(refine_layout)
        assert "artifact_id" in sig.parameters
        assert "feedback" in sig.parameters
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/tools/test.py::TestRefineLayout -v`
Expected: FAIL with ImportError

**Step 3: Create refine.py**

Create `src/mcp/tools/refine.py`:

```python
"""Refine layout tool for iterative improvements."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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

    result["parent_id"] = artifact_id
    return result


__all__ = ["refine_layout"]
```

**Step 4: Update __init__.py**

Add to `src/mcp/tools/__init__.py`:

```python
from .refine import refine_layout
```

**Step 5: Run tool tests**

Run: `pytest src/mcp/tools/test.py::TestRefineLayout -v`
Expected: PASS

**Step 6: Register in server.py**

```python
@mcp.tool
def refine_layout(
    artifact_id: str,
    feedback: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Refine an existing layout based on feedback.

    Use this when the user wants to MODIFY an existing layout
    rather than generate a completely new one.

    Args:
        artifact_id: ID of the layout to refine.
        feedback: What to change (e.g., "move sidebar to right").
        model: LLM model to use (optional).
        temperature: Creativity level 0.0-2.0. Default: 0.7

    Returns:
        Dictionary with new layout and parent_id link.

    Example:
        User: "I like this but move the sidebar to the right"
        → refine_layout(artifact_id, "move sidebar to right")
    """
    _validate_temperature(temperature)

    from .tools.refine import refine_layout as _refine

    return _refine(
        artifact_id=artifact_id,
        feedback=feedback,
        model=model,
        temperature=temperature,
    )
```

**Step 7: Update tool count to 10**

**Step 8: Run all tests and commit**

```bash
git add src/mcp/tools/refine.py src/mcp/tools/__init__.py src/mcp/server.py src/mcp/test.py
git commit -m "feat(mcp): add refine_layout tool for iterative feedback"
```

---

### Task 2.4: Make RAG Degradation Explicit

**Files:**
- Modify: `src/mcp/tools/generate.py`
- Add test to: `src/mcp/tools/test.py`

**Step 1: Write the failing test**

```python
class TestGenerateLayoutRAGStatus:
    """Tests for RAG status in generate_layout response."""

    @pytest.mark.unit
    def test_response_includes_rag_status(self, mock_generator_output):
        """Response includes explicit RAG availability status."""
        from src.mcp.tools.generate import generate_layout

        result = generate_layout(
            query="test",
            include_rag=True,
            persist=False,
        )

        # Should explicitly report RAG status
        assert "rag_status" in result
        assert "available" in result["rag_status"]
        assert "reason" in result["rag_status"]

    @pytest.mark.unit
    def test_rag_status_reports_unavailable_reason(self, monkeypatch):
        """RAG status explains why RAG is unavailable."""
        # Mock get_vector_store to return None
        monkeypatch.setattr(
            "src.mcp.tools.generate.get_vector_store",
            lambda: None
        )

        from src.mcp.tools.generate import generate_layout

        result = generate_layout(
            query="test",
            include_rag=True,
            persist=False,
        )

        assert result["rag_status"]["available"] is False
        assert "index" in result["rag_status"]["reason"].lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/tools/test.py::TestGenerateLayoutRAGStatus -v`
Expected: FAIL with KeyError: 'rag_status'

**Step 3: Add rag_status to generate_layout response**

Edit `src/mcp/tools/generate.py`, after line 130:

```python
# Add RAG status for explicit feedback
rag_status = {
    "available": rag_available,
    "requested": include_rag,
    "reason": (
        "RAG context included"
        if rag_available
        else "RAG index not found - run: python . dev index build"
    ),
}

result: dict[str, Any] = {
    "layout": layout_dict,
    "draft": draft,
    "rag_status": rag_status,  # Add this
    "stats": {
        # ... existing stats
    },
}
```

**Step 4: Run tests**

Run: `pytest src/mcp/tools/test.py::TestGenerateLayoutRAGStatus -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp/tools/generate.py src/mcp/tools/test.py
git commit -m "feat(mcp): add explicit rag_status to generate_layout response"
```

---

## Phase 3: Quality Measurement (3 Tasks)

### Task 3.1: Add Health Check Unit Tests

**Files:**
- Create: `src/mcp/health_test.py`

**Step 1: Write the tests**

Create `src/mcp/health_test.py`:

```python
"""Unit tests for health checking module."""

import pytest

from .health import (
    HealthStatus,
    ServiceStatus,
    check_kroki,
    check_llm_providers,
    check_rag_index,
    get_server_health,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    @pytest.mark.unit
    def test_status_values(self):
        """Health status has expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestServiceStatus:
    """Tests for ServiceStatus dataclass."""

    @pytest.mark.unit
    def test_available_service(self):
        """Available service has correct state."""
        status = ServiceStatus(
            available=True,
            message="Service running",
        )
        assert status.available is True
        assert status.message == "Service running"

    @pytest.mark.unit
    def test_unavailable_service(self):
        """Unavailable service has correct state."""
        status = ServiceStatus(
            available=False,
            message="Service not found",
            action="Start the service",
        )
        assert status.available is False
        assert status.action == "Start the service"


class TestCheckKroki:
    """Tests for check_kroki function."""

    @pytest.mark.unit
    def test_returns_service_status(self, monkeypatch):
        """check_kroki returns ServiceStatus."""
        # Mock RenderClient to avoid actual network call
        class MockClient:
            def is_available(self):
                return False

        monkeypatch.setattr(
            "src.mcp.health.RenderClient",
            lambda: MockClient()
        )

        result = check_kroki()

        assert isinstance(result, ServiceStatus)
        assert result.available is False


class TestCheckLLMProviders:
    """Tests for check_llm_providers function."""

    @pytest.mark.unit
    def test_returns_service_status(self):
        """check_llm_providers returns ServiceStatus."""
        result = check_llm_providers()

        assert isinstance(result, ServiceStatus)
        assert isinstance(result.available, bool)


class TestCheckRAGIndex:
    """Tests for check_rag_index function."""

    @pytest.mark.unit
    def test_returns_service_status(self):
        """check_rag_index returns ServiceStatus."""
        result = check_rag_index()

        assert isinstance(result, ServiceStatus)
        assert isinstance(result.available, bool)


class TestGetServerHealth:
    """Tests for get_server_health function."""

    @pytest.mark.unit
    def test_returns_complete_health(self):
        """get_server_health returns all required fields."""
        health = get_server_health()

        assert hasattr(health, "status")
        assert hasattr(health, "can_generate")
        assert hasattr(health, "can_preview")
        assert hasattr(health, "can_use_rag")

    @pytest.mark.unit
    def test_to_dict_format(self):
        """Health converts to expected dict format."""
        health = get_server_health()
        result = health.to_dict()

        assert "status" in result
        assert "version" in result
        assert "capabilities" in result
        assert "services" in result

    @pytest.mark.unit
    def test_capabilities_match_services(self):
        """Capabilities reflect actual service status."""
        health = get_server_health()
        result = health.to_dict()

        # Capabilities should match service availability
        assert result["capabilities"]["preview_layout"] == result["services"]["kroki"]["available"]
        assert result["capabilities"]["rag_context"] == result["services"]["rag_index"]["available"]
```

**Step 2: Run tests**

Run: `pytest src/mcp/health_test.py -v`
Expected: Most PASS, some may need health.py adjustments

**Step 3: Fix any failing tests by updating health.py**

**Step 4: Commit**

```bash
git add src/mcp/health_test.py
git commit -m "test(mcp): add health check unit tests"
```

---

### Task 3.2: Wire Audit Framework to Pytest

**Files:**
- Create: `src/vector/audit/test.py`
- Modify: `pytest.ini` (add rag_quality marker)

**Step 1: Write the test that uses audit framework**

Create `src/vector/audit/test.py`:

```python
"""Tests that exercise the RAG audit framework."""

import pytest

from .lib import AuditRunner, AuditConfig
from .metrics import MetricsCalculator, SearchMetrics


class TestMetricsCalculator:
    """Tests for metrics calculation."""

    @pytest.mark.unit
    def test_precision_at_k(self):
        """Precision@K calculates correctly."""
        calc = MetricsCalculator()

        # 3 relevant out of 5 retrieved
        relevant = {"a", "b", "c"}
        retrieved = ["a", "x", "b", "y", "z"]

        precision = calc.precision_at_k(retrieved, relevant, k=5)
        assert precision == 0.4  # 2/5

    @pytest.mark.unit
    def test_recall_at_k(self):
        """Recall@K calculates correctly."""
        calc = MetricsCalculator()

        relevant = {"a", "b", "c", "d"}
        retrieved = ["a", "b", "x"]

        recall = calc.recall_at_k(retrieved, relevant, k=3)
        assert recall == 0.5  # 2/4


class TestAuditConfig:
    """Tests for audit configuration."""

    @pytest.mark.unit
    def test_default_config(self):
        """Default config has sensible values."""
        config = AuditConfig()

        assert config.k_values == [1, 3, 5, 10]
        assert config.sample_size > 0


@pytest.mark.rag
@pytest.mark.integration
class TestAuditRunner:
    """Integration tests for audit runner."""

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return [
            ("login form", ["input", "button"]),
            ("dashboard", ["container", "card"]),
            ("navigation", ["navbar", "button"]),
        ]

    def test_runner_executes(self, sample_queries, vector_store):
        """Audit runner can execute on sample queries."""
        if vector_store is None:
            pytest.skip("RAG index not available")

        config = AuditConfig(sample_size=len(sample_queries))
        runner = AuditRunner(vector_store, config)

        results = runner.run(sample_queries)

        assert results is not None
        assert "precision" in results
        assert "recall" in results
```

**Step 2: Run tests**

Run: `pytest src/vector/audit/test.py -v`
Expected: Unit tests PASS, integration tests may SKIP

**Step 3: Commit**

```bash
git add src/vector/audit/test.py
git commit -m "test(vector): wire audit framework to pytest"
```

---

### Task 3.3: Create RAG A/B Benchmark Test

**Files:**
- Create: `tests/benchmarks/test_rag_ab.py`

**Step 1: Write the benchmark test**

Create `tests/benchmarks/test_rag_ab.py`:

```python
"""A/B benchmark comparing RAG vs non-RAG generation."""

import pytest
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Results from a single generation."""
    query: str
    rag_enabled: bool
    node_count: int
    max_depth: int
    component_types: set[str]
    generation_time_ms: float


BENCHMARK_QUERIES = [
    "login form with email and password",
    "dashboard with sidebar and stats",
    "settings page with toggle switches",
    "e-commerce product card",
    "navigation bar with menu items",
]


@pytest.mark.slow
@pytest.mark.rag
class TestRAGABBenchmark:
    """A/B benchmark for RAG vs non-RAG generation quality."""

    @pytest.fixture
    def generator(self, mock_llm_backend):
        """Generator configured for testing."""
        from src.llm import LayoutGenerator, GeneratorConfig

        return LayoutGenerator(
            backend=mock_llm_backend,
            config=GeneratorConfig(max_retries=1),
        )

    def test_rag_vs_no_rag_component_coverage(self, generator, vector_store):
        """RAG should improve component type coverage."""
        if vector_store is None:
            pytest.skip("RAG index required")

        results_with_rag = []
        results_without_rag = []

        for query in BENCHMARK_QUERIES[:3]:  # Limit for test speed
            # Generate with RAG
            generator.vector_store = vector_store
            output_rag = generator.generate(query)
            results_with_rag.append(self._analyze(query, True, output_rag))

            # Generate without RAG
            generator.vector_store = None
            output_no_rag = generator.generate(query)
            results_without_rag.append(self._analyze(query, False, output_no_rag))

        # Compare results
        avg_types_rag = sum(len(r.component_types) for r in results_with_rag) / len(results_with_rag)
        avg_types_no_rag = sum(len(r.component_types) for r in results_without_rag) / len(results_without_rag)

        # RAG should provide at least equal component variety
        # (This is a baseline - actual improvement TBD)
        assert avg_types_rag >= avg_types_no_rag * 0.9

    def _analyze(self, query: str, rag: bool, output) -> BenchmarkResult:
        """Analyze generation output."""
        node = output.context.node

        def count_nodes(n, depth=0):
            count = 1
            max_d = depth
            for child in n.children:
                c, d = count_nodes(child, depth + 1)
                count += c
                max_d = max(max_d, d)
            return count, max_d

        def get_types(n):
            types = {n.type.value}
            for child in n.children:
                types.update(get_types(child))
            return types

        count, max_depth = count_nodes(node)
        types = get_types(node)

        return BenchmarkResult(
            query=query,
            rag_enabled=rag,
            node_count=count,
            max_depth=max_depth,
            component_types=types,
            generation_time_ms=0,  # TODO: measure
        )
```

**Step 2: Run benchmark**

Run: `pytest tests/benchmarks/test_rag_ab.py -v --tb=short`
Expected: SKIP if RAG not available, otherwise provides comparison data

**Step 3: Commit**

```bash
git add tests/benchmarks/test_rag_ab.py
git commit -m "test(benchmark): add RAG A/B comparison benchmark"
```

---

## Phase 4: Value Validation (2 Tasks)

### Task 4.1: Add next_steps to Tool Responses

**Files:**
- Modify: `src/mcp/server.py`
- Modify: `src/mcp/test.py`

**Step 1: Write the failing test**

Add to `src/mcp/test.py`:

```python
class TestNextStepsGuidance:
    """Tests for next_steps in tool responses."""

    @pytest.mark.unit
    def test_status_includes_next_steps(self):
        """status() includes next_steps guidance."""
        from .server import status

        result = status.fn()

        assert "next_steps" in result
        assert isinstance(result["next_steps"], list)
        assert len(result["next_steps"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest src/mcp/test.py::TestNextStepsGuidance -v`
Expected: FAIL

**Step 3: Add next_steps to status() response**

Edit `src/mcp/server.py` status function, before return:

```python
# Add next steps guidance
if health.status == HealthStatus.HEALTHY:
    next_steps = [
        "Ready! Call generate_layout(query) to create a wireframe.",
        "Example: generate_layout('login form with email and password')",
    ]
elif health.can_generate:
    next_steps = [
        "Generation available. Preview may be limited.",
        "Call generate_layout(query) to create a wireframe.",
    ]
else:
    next_steps = [
        "Server not ready. Review action_required items above.",
        "Most common: Set OPENAI_API_KEY in .env file.",
    ]

result["next_steps"] = next_steps
```

**Step 4: Run tests and commit**

```bash
git add src/mcp/server.py src/mcp/test.py
git commit -m "feat(mcp): add next_steps guidance to status response"
```

---

### Task 4.2: Update SERVER_INSTRUCTIONS with Complete Workflow

**Files:**
- Modify: `src/mcp/server.py`

**Step 1: Update SERVER_INSTRUCTIONS**

Replace the SERVER_INSTRUCTIONS constant with updated version:

```python
SERVER_INSTRUCTIONS = """\
## Wireframe MCP Server

Generates UI wireframe layouts from natural language. Prevents implementation
miscommunication by validating understanding BEFORE coding begins.

### Quick Start
1. `status()` → check readiness
2. `generate_layout("your UI description")` → get draft
3. Review draft with user
4. `preview_layout(layout)` → visual wireframe (optional)
5. `validate_layout(layout)` → check before implementation

### Complete Workflow

**Creating Layouts:**
- `generate_layout(query)` - Single layout from description
- `generate_variations(query, count=3)` - Multiple options to compare
- `refine_layout(artifact_id, feedback)` - Iterate on existing layout

**Reviewing:**
- `get_artifact(id)` - Get full layout from artifact ID
- `search_layouts(query)` - Find similar examples in corpus
- `validate_layout(layout)` - Check for issues

**Preview:**
- `preview_layout(layout)` - Render visual wireframe

### Iteration Example
```
User: "Create a dashboard"
→ generate_layout("dashboard with sidebar")
→ User: "Make the sidebar narrower"
→ refine_layout(artifact_id, "make sidebar narrower")
→ User: "Perfect, show me the wireframe"
→ preview_layout(layout)
→ User: "Approved!"
→ validate_layout(layout)  # Check before implementation
```

### Service Status
| Service | Required For | If Missing |
|---------|-------------|------------|
| LLM provider | Generation | Set API key in .env |
| Kroki | Preview | Run: python . docker up |
| RAG index | Context | Run: python . dev index build |

### Getting Help
- `help()` - List topics
- `help('workflow')` - Step-by-step guide
- `help('troubleshooting')` - Common issues
"""
```

**Step 2: Commit**

```bash
git add src/mcp/server.py
git commit -m "docs(mcp): update SERVER_INSTRUCTIONS with complete workflow"
```

---

## Summary: Implementation Order

| Phase | Task | Description | Files Changed |
|-------|------|-------------|---------------|
| 1.1 | Fix crash bug | prompt_context.examples → example_count | generate.py, test.py |
| 1.2 | Register get_artifact | Enable variations workflow | server.py, test.py |
| 1.3 | Add MCP conftest | Proper test fixtures | conftest.py, test.py |
| 2.1 | Register validate_layout | Pre-implementation checks | server.py, test.py |
| 2.2 | Register search_layouts | Corpus exploration | server.py, test.py |
| 2.3 | Add refine_layout | Iterative feedback | refine.py, server.py |
| 2.4 | RAG status explicit | Clear degradation feedback | generate.py |
| 3.1 | Health check tests | Verify health reporting | health_test.py |
| 3.2 | Wire audit framework | Enable quality measurement | audit/test.py |
| 3.3 | RAG A/B benchmark | Measure RAG value | benchmarks/ |
| 4.1 | Add next_steps | Guide users through workflow | server.py |
| 4.2 | Update instructions | Complete workflow docs | server.py |

**Total: 12 tasks across 4 phases**

---

## Verification Commands

After each phase, run:

```bash
# Phase 1
pytest src/mcp/tools/test.py src/mcp/test.py -v -m unit

# Phase 2
pytest src/mcp/ -v -m unit

# Phase 3
pytest src/mcp/ src/vector/audit/ -v -m "unit or rag"

# Phase 4
pytest src/mcp/test.py -v

# Full suite
pytest src/mcp/ -v
```
