# Wireframe-MCP Implementation Task Tracker

> **Last Updated**: 2026-01-18
> **Branch**: feature-investigate
> **Status**: PLANNING

---

## Executive Summary

This document tracks the phased implementation of missing MCP functionality in the wireframe-mcp project. The project has complete CLI/library functionality but lacks its namesake MCP server implementation.

### Gap Categories

| Category | Priority | Status |
|----------|----------|--------|
| **Critical: MCP Server** | P0 | NOT STARTED |
| **Critical: MCP Tools** | P0 | NOT STARTED |
| **Critical: MCP Testing** | P0 | NOT STARTED |
| **Critical: Agentic Mode** | P1 | NOT STARTED |
| **Partial: WebUI Provider** | P2 | NOT STARTED |
| **Partial: Multi-Provider Tests** | P2 | NOT STARTED |

---

## Phase Overview

```
PHASE 1: MCP Server Core          [NOT STARTED] ░░░░░░░░░░ 0%
PHASE 2: MCP Tools & Resources    [NOT STARTED] ░░░░░░░░░░ 0%
PHASE 3: MCP Testing Framework    [NOT STARTED] ░░░░░░░░░░ 0%
PHASE 4: Agentic Mode             [NOT STARTED] ░░░░░░░░░░ 0%
PHASE 5: Partial Implementations  [NOT STARTED] ░░░░░░░░░░ 0%
```

---

## PHASE 1: MCP Server Core

**Goal**: Establish FastMCP server foundation with basic lifecycle management.

**Duration**: Foundation sprint

### Tasks

- [ ] **1.1** Add FastMCP dependency to pyproject.toml
  - Add `fastmcp>=2.0,<3` to dependencies
  - Add dev dependencies for MCP testing
  - File: `pyproject.toml`

- [ ] **1.2** Create MCP server module structure
  - Create `src/mcp/` directory
  - Create `src/mcp/__init__.py` with public exports
  - Create `src/mcp/server.py` with FastMCP instance
  - Create `src/mcp/lib.py` for core logic

- [ ] **1.3** Implement server lifecycle
  - STDIO transport (for Claude Desktop)
  - HTTP transport (for web deployment)
  - Graceful shutdown handling
  - Health check endpoint

- [ ] **1.4** Integrate with CLI entry point
  - Add `mcp` command to `__main__.py`
  - Support `python . mcp run` for STDIO mode
  - Support `python . mcp serve --port 18080` for HTTP mode

- [ ] **1.5** Create server configuration
  - Environment variable support (MCP_PORT, MCP_HOST)
  - Integration with existing `src/config/` module
  - Default values aligned with Docker compose

### Acceptance Criteria

- [ ] `python . mcp run` starts STDIO server
- [ ] `python . mcp serve` starts HTTP server on port 18080
- [ ] Server responds to MCP ping requests
- [ ] Server lists available tools (even if empty initially)

### Files to Create/Modify

```
src/mcp/
├── __init__.py          # Public API exports
├── lib.py               # Core MCP logic
├── server.py            # FastMCP instance and configuration
└── test.py              # Unit tests

__main__.py              # Add mcp command
pyproject.toml           # Add fastmcp dependency
```

---

## PHASE 2: MCP Tools & Resources

**Goal**: Expose existing CLI functionality as MCP tools and resources.

**Duration**: Feature sprint

### Tasks

- [ ] **2.1** Implement `generate_layout` tool
  - Wraps existing `LayoutGenerator.generate()`
  - Input: query (str), model (optional), temperature (optional)
  - Output: JSON layout, DSL code, stats
  - File: `src/mcp/tools/generate.py`

- [ ] **2.2** Implement `search_layouts` tool
  - Wraps existing `VectorStore.search()`
  - Input: query (str), k (int, default=5)
  - Output: list of similar layouts with scores
  - File: `src/mcp/tools/search.py`

- [ ] **2.3** Implement `render_layout` tool
  - Wraps existing `RenderClient.render()`
  - Input: layout (JSON or DSL), provider (d2/plantuml), format (png/svg)
  - Output: base64 encoded image data
  - File: `src/mcp/tools/render.py`

- [ ] **2.4** Implement `validate_layout` tool
  - Wraps existing `validate_layout()` from MID layer
  - Input: layout JSON
  - Output: validation result with errors
  - File: `src/mcp/tools/validate.py`

- [ ] **2.5** Implement `transpile_layout` tool
  - Wraps existing provider transpilers
  - Input: layout JSON, provider (d2/plantuml)
  - Output: DSL code string
  - File: `src/mcp/tools/transpile.py`

- [ ] **2.6** Create MCP resources
  - `schema://components` - Component type catalog
  - `schema://layout` - LayoutNode JSON schema
  - `config://models` - Available LLM models
  - `config://providers` - Available corpus providers
  - File: `src/mcp/resources.py`

- [ ] **2.7** Register all tools and resources
  - Wire up to FastMCP server instance
  - Add proper docstrings for LLM consumption
  - File: `src/mcp/server.py`

### Acceptance Criteria

- [ ] All 5 tools appear in MCP tool list
- [ ] `generate_layout` produces valid layouts
- [ ] `search_layouts` returns relevant results
- [ ] `render_layout` produces images
- [ ] Resources are accessible via MCP protocol

### Files to Create

```
src/mcp/
├── tools/
│   ├── __init__.py
│   ├── generate.py
│   ├── search.py
│   ├── render.py
│   ├── validate.py
│   └── transpile.py
├── resources.py
└── server.py            # Updated with registrations
```

---

## PHASE 3: MCP Testing Framework

**Goal**: Comprehensive test coverage for MCP functionality.

**Duration**: Quality sprint

### Tasks

- [ ] **3.1** Create MCP test fixtures
  - FastMCP test client fixture
  - Mock LLM backend fixture
  - Test vector index fixture
  - File: `src/mcp/conftest.py`

- [ ] **3.2** Add pytest marker for MCP tests
  - Add `mcp` marker to pytest.ini
  - Configure marker behavior
  - File: `pytest.ini`

- [ ] **3.3** Unit tests for each tool
  - Test input validation
  - Test error handling
  - Test output format
  - Files: `src/mcp/tools/test.py`

- [ ] **3.4** Integration tests with MCP client
  - Test tool invocation via protocol
  - Test resource access
  - Test concurrent requests
  - File: `src/mcp/test_integration.py`

- [ ] **3.5** Agent simulation tests
  - Simulate Claude Desktop calling tools
  - Test multi-tool workflows
  - Test error recovery
  - File: `src/mcp/test_agent.py`

- [ ] **3.6** Performance benchmarks
  - Measure tool response times
  - Measure concurrent request handling
  - File: `src/mcp/test_perf.py`

### Acceptance Criteria

- [ ] `pytest -m mcp` runs all MCP tests
- [ ] >90% code coverage for MCP module
- [ ] Integration tests pass with real MCP client
- [ ] Agent simulation tests validate workflows

### Files to Create

```
src/mcp/
├── conftest.py          # Test fixtures
├── test.py              # Unit tests
├── test_integration.py  # Integration tests
├── test_agent.py        # Agent simulation
└── test_perf.py         # Performance tests

pytest.ini               # Add mcp marker
```

---

## PHASE 4: Agentic Mode

**Goal**: Server-side LLM orchestration for multi-turn conversations.

**Duration**: Advanced sprint

### Tasks

- [ ] **4.1** Design conversation state model
  - Define conversation context schema
  - Session management
  - File: `src/mcp/agent/state.py`

- [ ] **4.2** Implement tool chaining
  - Allow generate → validate → render flow
  - Track intermediate results
  - File: `src/mcp/agent/chain.py`

- [ ] **4.3** Implement iterative refinement
  - User feedback loop
  - Layout modification based on feedback
  - File: `src/mcp/agent/refine.py`

- [ ] **4.4** Create agent tool
  - MCP tool for starting agent sessions
  - Support multi-turn interactions
  - File: `src/mcp/tools/agent.py`

- [ ] **4.5** Add agent tests
  - Test conversation flow
  - Test state persistence
  - Test refinement cycles
  - File: `src/mcp/agent/test.py`

### Acceptance Criteria

- [ ] Agent can maintain conversation context
- [ ] Tool chaining works for common workflows
- [ ] Iterative refinement produces improved layouts
- [ ] Agent sessions are properly managed

### Files to Create

```
src/mcp/
├── agent/
│   ├── __init__.py
│   ├── state.py
│   ├── chain.py
│   ├── refine.py
│   └── test.py
└── tools/
    └── agent.py
```

---

## PHASE 5: Partial Implementations

**Goal**: Complete remaining partial implementations.

**Duration**: Completion sprint

### Tasks

- [ ] **5.1** Implement WebUI corpus provider
  - Create `src/corpus/provider/webui/` module
  - Implement data fetching and normalization
  - Add tests
  - File: `src/corpus/provider/webui/lib.py`

- [ ] **5.2** Add multi-provider corpus tests
  - Test combining multiple providers
  - Test deduplication
  - Test cross-provider search
  - File: `src/corpus/test_multi.py`

- [ ] **5.3** Add E2E LLM integration tests
  - Tests with real LLM calls (optional, slow)
  - Mark with `@pytest.mark.slow`
  - File: `src/tests/test_e2e_llm.py`

- [ ] **5.4** Documentation updates
  - Update README.md status table
  - Add MCP usage documentation
  - Add API reference

### Acceptance Criteria

- [ ] WebUI provider fetches and normalizes data
- [ ] Multi-provider tests pass
- [ ] E2E tests work with real LLMs
- [ ] Documentation is current

---

## Progress Log

### 2026-01-18 - Initial Planning

- Created task.md tracking document
- Analyzed codebase for implementation gaps
- Defined 5-phase implementation plan
- Created supporting research documents

---

## Commit Checkpoints

Use these as commit points during implementation:

1. **Phase 1 Complete**: `feat(mcp): add MCP server core with STDIO/HTTP transport`
2. **Phase 2.1-2.2**: `feat(mcp): add generate_layout and search_layouts tools`
3. **Phase 2.3-2.5**: `feat(mcp): add render, validate, and transpile tools`
4. **Phase 2.6-2.7**: `feat(mcp): add MCP resources and complete tool registration`
5. **Phase 3 Complete**: `test(mcp): add comprehensive MCP test suite`
6. **Phase 4 Complete**: `feat(mcp): add agentic mode with tool chaining`
7. **Phase 5 Complete**: `feat(corpus): add WebUI provider and multi-provider tests`

---

## Dependencies

### External Dependencies to Add

```toml
# pyproject.toml additions
dependencies = [
    "fastmcp>=2.0,<3",
    # existing deps...
]

[project.optional-dependencies]
dev = [
    "pytest-asyncio>=0.23",
    # existing dev deps...
]
```

### Internal Dependencies

- Phase 2 depends on Phase 1 (server must exist for tools)
- Phase 3 depends on Phase 2 (tools must exist for testing)
- Phase 4 depends on Phase 2 (tools must exist for chaining)
- Phase 5 is independent

---

## Notes

- All MCP tools should have comprehensive docstrings for LLM consumption
- Tools should return structured JSON, not plain text
- Error messages should be actionable and specific
- Performance matters - tools should respond quickly
