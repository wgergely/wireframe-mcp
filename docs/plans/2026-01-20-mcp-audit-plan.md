# MCP Server Comprehensive Audit Plan

**Date**: 2026-01-20
**Status**: Planning
**Goal**: Validate that wireframe-mcp delivers on its value proposition - preventing expensive implementation mistakes by validating wireframe understanding BEFORE LLM coding agents begin work.

---

## Executive Summary

The wireframe-mcp server aims to bridge communication between human designers and LLM coding agents through:

```
Natural Language → [Draft] ↔ User Feedback → [Approved Draft] → Transpilation + RAG → Visual Wireframe
```

**Critical Question**: Does this architecture actually prevent miscommunication, or does it add complexity without measurable benefit?

### Key Findings from Initial Analysis

| Area | Status | Severity |
|------|--------|----------|
| RAG value unmeasured | No A/B testing, audit framework orphaned | **Critical** |
| MCP tools incomplete | 8 implemented, 6 exposed, workflow broken | **Critical** |
| Test infrastructure | Mocks hide failures, MCP tests can't import | **High** |
| User feedback loop | Draft preview exists, but no iteration tools | **High** |
| Service dependencies | Inconsistent degradation, silent failures | **Medium** |

---

## Part 1: Implementation Correctness Audit

### 1.1 RAG Value Proposition (Priority: Critical)

**Hypothesis to Test**: RAG-backed generation produces layouts that better match user intent than pure LLM generation.

#### Current State
- RAG injects 1-3 similar layouts as few-shot examples in prompt
- Similar layouts **never returned to user** (TODO in code, always empty list)
- Audit framework exists (`src/vector/audit/`) but **never called**
- No metrics comparing RAG vs non-RAG output quality

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| R1 | Fix `prompt_context.examples` crash bug | Code review | `generate.py:128-129` uses correct field |
| R2 | Return similar layouts to user | Implementation | `TranspilationContext.similar_layouts` populated |
| R3 | Create RAG A/B benchmark | Testing | 20+ queries with/without RAG, measure structural similarity |
| R4 | Define "layout quality" metrics | Design | Documented criteria: component appropriateness, hierarchy depth, semantic match |
| R5 | Wire audit framework to test suite | Integration | `pytest -m rag` runs precision/recall benchmarks |

#### Key Questions
- Does RAG context improve component selection for domain-specific UIs (e-commerce vs dashboard vs form)?
- Is the corpus (Rico, WebSight, Enrico) representative of modern UI patterns?
- What's the latency cost of RAG retrieval vs quality benefit?

---

### 1.2 MCP Tool Completeness (Priority: Critical)

**Hypothesis to Test**: The exposed MCP tools support the complete user feedback workflow.

#### Current State
```
Implemented: 14 tools
Registered:   6 tools (generate_layout, preview_layout, generate_variations, status, help, list_models)
Orphaned:     8 tools (validate_layout, search_layouts, transpile_layout, get_history, get_artifact, get_sessions, get_variation_set, cleanup_history)
```

#### Workflow Gap Analysis

**Intended Workflow**:
```
1. User: "Create a dashboard with sidebar navigation"
2. Agent: generate_layout() → returns draft + artifact_id
3. User: "Show me similar examples from the corpus"
4. Agent: search_layouts() → NOT EXPOSED ❌
5. User: "I like variation 2, preview it"
6. Agent: get_artifact(variation_id) → NOT EXPOSED ❌
7. Agent: preview_layout(layout) → returns image
8. User: "Approved, validate before implementation"
9. Agent: validate_layout() → NOT EXPOSED ❌
```

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| T1 | Document intended vs actual workflow | Analysis | Gap matrix showing blocked user journeys |
| T2 | Register `get_artifact` tool | Implementation | Variations workflow unblocked |
| T3 | Register `validate_layout` tool | Implementation | Pre-implementation validation available |
| T4 | Register `search_layouts` tool | Implementation | Corpus exploration enabled |
| T5 | Delete or document orphaned tools | Decision | No dead code in production API |
| T6 | Add iteration/refinement tool | Design | `refine_layout(artifact_id, feedback)` for feedback loop |

#### Key Questions
- Why were 8 tools implemented but explicitly excluded in tests?
- Is the 6-tool surface intentional minimalism or incomplete work?
- Does the LLM coding agent need history access, or is that human-only?

---

### 1.3 Draft ↔ Feedback Loop (Priority: High)

**Hypothesis to Test**: The draft representation enables effective human feedback before visual rendering.

#### Current State
- `generate_layout()` returns `draft` (text tree) + `layout` (JSON)
- No tool for "refine this draft based on my feedback"
- Variations require re-generating from scratch
- Parent/child linking exists but no tool exposes it

#### Draft Format Analysis

```
# Current draft output (text tree):
container[main] horizontal
├── navbar[nav] vertical
│   └── text[title]: "Dashboard"
├── sidebar[side] vertical
│   └── list_item[menu]: "Settings"
└── container[content] vertical
    └── card[widget]: "Stats"
```

**Questions**:
- Is this representation sufficient for user feedback?
- Can users say "move sidebar to right" and agent understand?
- Should draft include visual hints (dimensions, spacing)?

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| F1 | Test draft comprehension | User study (5 samples) | Users can describe changes from draft |
| F2 | Implement `refine_layout` tool | Implementation | Takes artifact_id + natural language feedback |
| F3 | Add parent linking to variations | Implementation | `parent_id` creates refinement chain |
| F4 | Expose refinement history | Tool registration | `get_lineage(artifact_id)` shows evolution |

---

### 1.4 Transpilation + Preview Pipeline (Priority: High)

**Hypothesis to Test**: The D2/PlantUML transpilation produces accurate visual representations of the layout.

#### Current State
- D2 and PlantUML providers implemented (`src/providers/`)
- Kroki service renders diagrams to PNG/SVG
- `preview_layout()` works but requires Kroki running
- No validation that transpiled output matches input layout

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| P1 | Transpilation fidelity test | Visual comparison | 10 layouts: JSON → D2 → render → manual check |
| P2 | Test deep nesting (5+ levels) | Stress test | Complex layouts render without truncation |
| P3 | Test all 26 component types | Coverage | Each ComponentType has visual representation |
| P4 | Add transpilation validation | Implementation | Post-render check that component count matches |
| P5 | Document Kroki fallback | Design | What happens when Kroki unavailable? |

---

## Part 2: User Feedback Quality Audit

### 2.1 Error Messages and Guidance (Priority: High)

#### Current State
- `status()` tool reports health but not actionable next steps
- Generation failures propagate as raw exceptions
- Silent degradation when RAG unavailable

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| U1 | Catalog all error messages | Code review | List every user-facing error |
| U2 | Add actionable guidance to errors | Implementation | Each error includes "try this instead" |
| U3 | Make RAG degradation explicit | Implementation | Response includes `rag_available: false, reason: "..."` |
| U4 | Test error message clarity | User study | Users understand what went wrong |

---

### 2.2 Progressive Disclosure (Priority: Medium)

#### Current State
- `help()` tool exists with topics
- `list_models()` shows available providers
- No guidance on "what should I do next?"

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| D1 | Add workflow suggestions to responses | Implementation | Each tool response includes `next_steps: [...]` |
| D2 | Add quality indicators to drafts | Implementation | `confidence: 0.8, similar_found: 3` |
| D3 | Surface RAG match quality | Implementation | Show which corpus examples influenced output |

---

## Part 3: Test Infrastructure Audit

### 3.1 Mock Quality (Priority: Critical)

#### Current State
- `MockLLMBackend` has 4 hardcoded responses - never fails, never rate-limits
- Network/IO failures completely untested
- MCP tests cannot import (`fastmcp` missing from environment)

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| M1 | Fix fastmcp import | Environment | `pytest -m mcp` collects tests |
| M2 | Add realistic LLM mock | Implementation | Mock that fails 10% of time, rate-limits |
| M3 | Add network failure tests | Implementation | Test download timeout, partial file, corrupted archive |
| M4 | Add Kroki failure tests | Implementation | Test service unavailable, timeout, malformed diagram |
| M5 | Remove silent test skipping | Configuration | Tests fail explicitly if service unavailable |

---

### 3.2 Integration Test Coverage (Priority: High)

#### Current State
- E2E tests call real LLM (expensive, non-deterministic)
- No contract tests between components
- Health check logic (`src/mcp/health.py`) completely untested

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| I1 | Add health check unit tests | Implementation | All 4 check functions tested |
| I2 | Add contract tests for tool interfaces | Implementation | Mock MCP client verifies response shapes |
| I3 | Add deterministic E2E tests | Implementation | Use recorded LLM responses |
| I4 | Test graceful degradation paths | Implementation | Each service failure tested |

---

### 3.3 RAG Quality Tests (Priority: High)

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| Q1 | Wire audit framework to pytest | Integration | `pytest -m rag` runs quality metrics |
| Q2 | Create golden query set | Data | 20 queries with expected component types |
| Q3 | Benchmark embedding models | Comparison | Voyage vs local, measure retrieval quality |
| Q4 | Test index staleness detection | Implementation | Warn if index older than corpus |

---

## Part 4: Architecture Value Audit

### 4.1 MCP vs Alternatives (Priority: Medium)

#### Questions to Answer
- Does MCP add value over HTTP REST API for this use case?
- Is STDIO transport appropriate for Claude Desktop integration?
- Should tools be stateless or maintain conversation context?

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| A1 | Compare MCP vs REST latency | Benchmark | Measure round-trip times |
| A2 | Evaluate streaming potential | Analysis | Can generation stream partial results? |
| A3 | Document transport trade-offs | Analysis | STDIO vs HTTP vs SSE for each use case |

---

### 4.2 Dependency Management (Priority: Medium)

#### Current State
- Kroki: Required for preview, Docker-based
- RAG: Optional but silently degrades
- LLM: Required, multiple providers

#### Audit Tasks

| ID | Task | Method | Success Criteria |
|----|------|--------|------------------|
| D1 | Document minimum viable configuration | Documentation | What's needed for basic workflow? |
| D2 | Add dependency health dashboard | Implementation | `status()` shows actionable setup steps |
| D3 | Test offline/degraded modes | Integration | Each degradation path documented and tested |

---

## Audit Execution Plan

### Phase 1: Critical Bugs (Week 1)
- [ ] R1: Fix `prompt_context.examples` crash
- [ ] M1: Fix fastmcp import for tests
- [ ] T2: Register `get_artifact` tool

### Phase 2: Workflow Completion (Week 2)
- [ ] T3-T5: Register remaining workflow tools
- [ ] F2: Implement `refine_layout` tool
- [ ] U3: Make degradation explicit

### Phase 3: Quality Measurement (Week 3)
- [ ] R3: Create RAG A/B benchmark
- [ ] Q1-Q2: Wire audit framework, create golden queries
- [ ] I1-I2: Add health check and contract tests

### Phase 4: Value Validation (Week 4)
- [ ] R4: Define layout quality metrics
- [ ] F1: User study on draft comprehension
- [ ] A1-A3: Architecture comparison

---

## Success Criteria

The audit is complete when we can answer:

1. **RAG Value**: "RAG improves layout quality by X% on [metric] for [query types]"
2. **Workflow**: "A user can go from NL description to approved wireframe in N tool calls"
3. **Reliability**: "Tests catch Y% of regressions with Z% confidence"
4. **User Feedback**: "Users understand draft output and can provide actionable feedback"

---

## Appendix: File References

### Critical Files to Review
- `src/mcp/server.py:124-499` - Tool registrations
- `src/mcp/tools/generate.py:128-129` - Crash bug
- `src/llm/generator/lib.py:283` - Empty similar_layouts TODO
- `src/vector/audit/` - Orphaned audit framework
- `src/mcp/test.py:162-169` - Explicit tool exclusions

### Existing Design Documents
- `temp/mcp-testing-strategy.md`
- `temp/mcp-workflow-design.md`
- `temp/agentic-mode-design.md`
- `temp/history-manager-design.md`
