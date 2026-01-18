# MCP Testing Strategy

> **Purpose**: Define testing approach for MCP server and tools
> **Last Updated**: 2026-01-18

---

## Overview

This document outlines the testing strategy for the MCP server implementation, covering unit tests, integration tests, and agent simulation tests.

---

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`)

Test individual tool logic in isolation without MCP protocol.

**Scope**:
- Tool input validation
- Business logic correctness
- Error handling
- Output format

**Example**:
```python
# src/mcp/tools/test.py
import pytest
from src.mcp.tools.generate import generate_layout_logic

@pytest.mark.unit
def test_generate_layout_returns_valid_structure():
    """Tool returns expected output structure."""
    result = generate_layout_logic(
        query="simple login form",
        model=None,
        temperature=0.7,
    )

    assert "layout" in result
    assert "stats" in result
    assert result["layout"]["type"] in ["container", "card"]

@pytest.mark.unit
def test_generate_layout_validates_temperature():
    """Tool rejects invalid temperature values."""
    with pytest.raises(ValueError, match="temperature"):
        generate_layout_logic(
            query="test",
            temperature=5.0,  # Invalid
        )
```

---

### 2. Integration Tests (`@pytest.mark.mcp`)

Test tools via MCP protocol with real client.

**Scope**:
- MCP protocol compliance
- Tool registration
- Resource access
- Client-server communication

**Example**:
```python
# src/mcp/test_integration.py
import pytest
from fastmcp import Client

@pytest.fixture
async def mcp_client():
    from src.mcp.server import create_server
    server = create_server()
    async with Client(server) as client:
        yield client

@pytest.mark.mcp
@pytest.mark.asyncio
async def test_tool_list(mcp_client):
    """Server exposes expected tools."""
    tools = await mcp_client.list_tools()
    tool_names = [t.name for t in tools]

    assert "generate_layout" in tool_names
    assert "search_layouts" in tool_names
    assert "validate_layout" in tool_names

@pytest.mark.mcp
@pytest.mark.asyncio
async def test_generate_tool(mcp_client):
    """generate_layout tool works via MCP."""
    result = await mcp_client.call_tool(
        "generate_layout",
        {"query": "login form"}
    )

    assert result.is_success
    assert "layout" in result.data

@pytest.mark.mcp
@pytest.mark.asyncio
async def test_resource_access(mcp_client):
    """Resources are accessible via MCP."""
    result = await mcp_client.read_resource("schema://components")

    assert result.is_success
    assert "components" in result.data
```

---

### 3. Agent Simulation Tests (`@pytest.mark.agent`)

Test realistic agent workflows and multi-tool sequences.

**Scope**:
- Multi-step workflows
- Tool chaining
- Error recovery
- State management

**Example**:
```python
# src/mcp/test_agent.py
import pytest
from fastmcp import Client

@pytest.mark.agent
@pytest.mark.asyncio
async def test_generate_validate_render_workflow(mcp_client):
    """Simulate: generate → validate → render workflow."""

    # Step 1: Generate layout
    gen_result = await mcp_client.call_tool(
        "generate_layout",
        {"query": "dashboard with sidebar and main content"}
    )
    assert gen_result.is_success
    layout = gen_result.data["layout"]

    # Step 2: Validate the generated layout
    val_result = await mcp_client.call_tool(
        "validate_layout",
        {"layout": layout}
    )
    assert val_result.is_success
    assert val_result.data["valid"] is True

    # Step 3: Render to image
    render_result = await mcp_client.call_tool(
        "render_layout",
        {"layout": layout, "format": "png"}
    )
    assert render_result.is_success
    assert "data" in render_result.data  # Base64 image

@pytest.mark.agent
@pytest.mark.asyncio
async def test_search_then_generate_workflow(mcp_client):
    """Simulate: search for examples → generate similar."""

    # Step 1: Search for similar layouts
    search_result = await mcp_client.call_tool(
        "search_layouts",
        {"query": "settings page with toggles", "k": 3}
    )
    assert search_result.is_success

    # Step 2: Generate with context from search
    gen_result = await mcp_client.call_tool(
        "generate_layout",
        {
            "query": "settings page similar to search results",
            "include_rag": True,
        }
    )
    assert gen_result.is_success

@pytest.mark.agent
@pytest.mark.asyncio
async def test_error_recovery_workflow(mcp_client):
    """Simulate: handle validation error and retry."""

    # Step 1: Generate (may have validation issues)
    gen_result = await mcp_client.call_tool(
        "generate_layout",
        {"query": "complex nested layout"}
    )

    if not gen_result.is_success:
        # Step 2: Check what went wrong
        error_msg = gen_result.error

        # Step 3: Retry with simpler query
        retry_result = await mcp_client.call_tool(
            "generate_layout",
            {"query": "simple container layout"}
        )
        assert retry_result.is_success
```

---

### 4. Performance Tests (`@pytest.mark.slow`)

Test response times and throughput.

**Scope**:
- Tool response latency
- Concurrent request handling
- Memory usage

**Example**:
```python
# src/mcp/test_perf.py
import pytest
import asyncio
import time

@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_response_time(mcp_client):
    """generate_layout responds within acceptable time."""
    start = time.monotonic()

    await mcp_client.call_tool(
        "generate_layout",
        {"query": "simple form"}
    )

    elapsed = time.monotonic() - start
    assert elapsed < 30.0, f"Generation took {elapsed:.1f}s"

@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_requests(mcp_client):
    """Server handles concurrent requests."""
    queries = [
        "login form",
        "dashboard",
        "settings page",
        "navigation bar",
        "user profile",
    ]

    start = time.monotonic()

    tasks = [
        mcp_client.call_tool("generate_layout", {"query": q})
        for q in queries
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.monotonic() - start

    # All should succeed
    assert all(r.is_success for r in results)

    # Should be faster than sequential
    assert elapsed < 60.0  # Less than 5 * 12s sequential

@pytest.mark.slow
@pytest.mark.asyncio
async def test_search_performance(mcp_client):
    """search_layouts responds quickly."""
    start = time.monotonic()

    for _ in range(10):
        await mcp_client.call_tool(
            "search_layouts",
            {"query": "navigation menu", "k": 5}
        )

    elapsed = time.monotonic() - start

    # 10 searches should complete in < 5 seconds
    assert elapsed < 5.0
```

---

## Test Fixtures

### conftest.py

```python
# src/mcp/conftest.py
import pytest
from pathlib import Path
from fastmcp import Client

from src.mcp.server import create_server


@pytest.fixture
def mcp_server():
    """Create MCP server instance for testing."""
    return create_server()


@pytest.fixture
async def mcp_client(mcp_server):
    """Create connected MCP client."""
    async with Client(mcp_server) as client:
        yield client


@pytest.fixture
def mock_llm_backend():
    """Mock LLM backend for unit tests."""
    from unittest.mock import MagicMock

    backend = MagicMock()
    backend.generate.return_value = MagicMock(
        content='{"id": "root", "type": "container", "children": []}',
        model="test-model",
        usage={"total_tokens": 100},
    )
    return backend


@pytest.fixture
def test_layout():
    """Sample layout for testing."""
    from src.mid import LayoutNode, Orientation
    from src.schema import ComponentType

    return LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        orientation=Orientation.VERTICAL,
        children=[
            LayoutNode(id="header", type=ComponentType.TOOLBAR),
            LayoutNode(id="content", type=ComponentType.CONTAINER),
        ],
    )


@pytest.fixture
def test_index(tmp_path):
    """Create temporary test index."""
    from src.vector import VectorStore, BackendType

    store = VectorStore(backend=BackendType.LOCAL)

    # Add some test items
    # ...

    index_path = tmp_path / "test_index"
    store.save(index_path)
    return index_path
```

---

## pytest.ini Updates

```ini
[pytest]
markers =
    unit: Fast unit tests with no external dependencies
    integration: Tests requiring file system or network
    docker: Tests requiring Docker services
    kroki: Tests requiring Kroki rendering service
    rag: Tests requiring vector index
    mcp: MCP protocol integration tests
    agent: Agent workflow simulation tests
    slow: Performance and stress tests
    gpu: Tests requiring GPU acceleration

asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
```

---

## Test Data

### Sample Queries

```python
TEST_QUERIES = [
    # Simple layouts
    "login form with email and password",
    "simple dashboard",
    "navigation bar",

    # Complex layouts
    "dashboard with sidebar, header, and content area",
    "settings page with sections and toggles",
    "e-commerce product listing with filters",

    # Edge cases
    "empty container",
    "deeply nested structure with 5 levels",
    "layout with all component types",
]
```

### Expected Outputs

```python
EXPECTED_OUTPUTS = {
    "login form": {
        "must_contain_types": ["input", "button"],
        "max_depth": 4,
    },
    "dashboard": {
        "must_contain_types": ["container"],
        "min_children": 2,
    },
    "navigation bar": {
        "must_contain_types": ["toolbar", "button"],
    },
}
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/mcp-tests.yml
name: MCP Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e ".[dev]"
      - run: pytest -m unit -v

  mcp-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e ".[dev]"
      - run: pytest -m mcp -v

  agent-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests, mcp-tests]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e ".[dev]"
      - run: pytest -m agent -v
```

---

## Coverage Requirements

| Category | Minimum Coverage |
|----------|------------------|
| Unit Tests | 90% |
| Integration Tests | 80% |
| Agent Tests | 70% |

### Coverage Commands

```bash
# Run all MCP tests with coverage
pytest -m "mcp or unit" --cov=src/mcp --cov-report=html

# Run specific test file
pytest src/mcp/test_integration.py -v

# Run agent tests only
pytest -m agent -v
```

---

## Test Execution Order

1. **Unit tests first** - Fast feedback, no dependencies
2. **MCP integration tests** - Protocol compliance
3. **Agent simulation tests** - Workflow validation
4. **Performance tests** - Only in CI or explicit runs

```bash
# Recommended test run sequence
pytest -m unit -q              # ~10 seconds
pytest -m mcp -q               # ~30 seconds
pytest -m agent -q             # ~2 minutes
pytest -m slow -q              # ~5 minutes (optional)
```
