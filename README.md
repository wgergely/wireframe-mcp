# Wireframe MCP: Layout Broker for AI-Assisted UI Design

## Vision

**Wireframe MCP** is a Model Context Protocol (MCP) server that bridges the gap between natural language UI descriptions and verified structural blueprints. It addresses a fundamental limitation in LLM-based UI generation: the "stochastic gap" where direct code generation leads to structural hallucinations and non-deterministic layouts.

Instead of letting LLMs generate raw HTML/React/CSS directly (where they often "imagine" impossible layouts), this system constrains LLM output to a strict Intermediate Representation (IR), validates it deterministically, and renders it to visual previews for human verification before any implementation code is written.

## The Problem We Solve

When a user asks an LLM to "make a dashboard with a sidebar and floating search", three critical issues emerge:

1. **The Black Box Effect**: No visibility into what the LLM "imagines" before code is written
2. **Spatial Blindness**: LLMs lack inherent understanding of containment, overlap, docking, and flex relationships
1.  **The Black Box Effect**: No visibility into what the LLM "imagines" before code is written
2.  **Spatial Blindness**: LLMs lack inherent understanding of containment, overlap, docking, and flex relationships
3.  **Wasted Iterations**: Structural errors are discovered only after implementation, requiring costly rewrites

Research (Chen et al. 2025, UIFormer Dec 2025) demonstrates that constraining LLM output to structured schemas improves user preference win rates by ~30% and reduces token consumption by ~50% while improving semantic accuracy.

## Architecture: The Compiler Pipeline

The system implements a **Three-Stage Compiler** pattern that transforms fuzzy natural language intent into rigid, validated syntax. 

For a deep dive into the data flow between **Corpus**, **MID**, and **IR**, see the [Architecture Guide](docs/architecture.md).

```
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 1: SEMANTIC PARSER (LLM)                │
│                                                                 │
│  • Input: Natural language ("dashboard with sidebar on left")   │
│  • Output: Structured JSON conforming to LayoutNode schema      │
│  • Constraint: JSON Schema forces syntactically valid trees     │
│  • Key: LLM fills a template, does NOT invent the syntax        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 2: MIDDLEWARE (Python MCP Server)           │
│                                                                 │
│  • Validates the AST (cycle detection, orphan removal)          │
│  • Type-checks properties (flex_ratio in 1-12 range)            │
│  • Enforces containment rules and accessibility constraints     │
│  • Rejects invalid structures with clear error messages         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 3: RENDERER (D2/PlantUML/Kroki)            │
│                                                                 │
│  • Deterministically transpiles validated IR to DSL             │
│  • Renders to PNG/SVG via Kroki for visual verification         │
│  • User approves OR requests changes (loop back to Stage 1)     │
└─────────────────────────────────────────────────────────────────┘
```

## The Intermediate Representation (IR)

The IR is the contract between the LLM and the runtime. It defines a restricted vocabulary of UI primitives based on the Rico dataset's 26 semantic categories:

### Component Types

| Category   | Types                                                                      |
|------------|----------------------------------------------------------------------------|
| Containers | `container`, `card`, `modal`, `web_view`                                   |
| Navigation | `toolbar`, `navbar`, `bottom_nav`, `drawer`, `tab_bar`, `multi_tab`        |
| Content    | `text`, `image`, `list_item`, `icon`, `advertisement`                      |
| Controls   | `button`, `input`, `checkbox`, `radio_button`, `switch`, `slider`, etc.    |

### Layout Model

```python
class LayoutNode:
    id: str                    # Unique identifier
    type: ComponentType        # From restricted vocabulary
    label: str | None          # Human-readable text
    flex_ratio: int            # 1-12 grid span (CSS flex basis)
    children: list[LayoutNode] # Recursive hierarchy
    orientation: Orientation   # horizontal | vertical | overlay
```

### Orientation Semantics

- **HORIZONTAL**: Children flow left-to-right (CSS `flex-direction: row`)
- **VERTICAL**: Children flow top-to-bottom (CSS `flex-direction: column`)
- **OVERLAY**: Children stack on z-axis (absolute positioning)

## Transpilation Providers

The validated IR is transpiled to provider-specific DSL syntax:

### D2 (Primary)

Selected for its **constraint-based layout engine** that explicitly models containers and connections, aligning 1:1 with CSS Flexbox/Grid systems.

```d2
root_container: Dashboard {
  direction: right

  sidebar: Navigation {
    width: 25%
  }

  main_content: Content Area {
    width: 75%
  }
}
```

### PlantUML Salt (Fallback)

Offers a sketchy, wireframe aesthetic with high tolerance for LLM output variations. Uses brace styles for layout hints:

- `{#` for grid layouts (horizontal containers)
- `{+` for tabbed panels (navbar/sidebar)
- `{` for simple grouping

## Current Implementation Status

### Completed

| Module              | Status | Description                                              |
|---------------------|--------|----------------------------------------------------------|
| `src/mid`           | Done   | Semantic source of truth (Models & Validation)           |
| `src/ir`            | Done   | Transpilation context (Bridge to Providers)              |
| `src/providers/d2`  | Done   | D2 DSL transpiler with direction/width hints             |
| `src/providers/plantuml` | Done | PlantUML Salt transpiler with component rendering      |
| `src/corpus`        | Done   | Rico dataset download and integration                    |

### In Progress / Planned

| Module              | Status | Description                                              |
|---------------------|--------|----------------------------------------------------------|
| MCP Server          | Planned     | FastMCP-based tool exposure                              |
| Kroki Integration   | Planned     | HTTP rendering to PNG/SVG                                |
| RAG System          | Planned     | FAISS-GPU vector search for layout retrieval             |
| Agentic Mode        | Planned     | Server-side LLM orchestration                            |

## Future: RAG-Enhanced Layout Generation

The system is designed to integrate Retrieval-Augmented Generation (RAG) for improved layout suggestions:

1. **Corpus**: Rico dataset (72k Android screens) with semantic annotations
2. **Vector Store**: FAISS-GPU for high-performance similarity search
3. **Workflow**: User query → Embed → Retrieve similar layouts → Few-shot context → LLM generation

This enables "style transfer" from existing premium designs and reduces hallucinations by grounding generation in real-world examples.

## Data Flow Example

### Input (Natural Language)
```
"I need a dashboard with a sidebar on the left and a main content area."
```

### Stage 1 Output (LLM → JSON IR)
```json
{
  "id": "root",
  "type": "container",
  "orientation": "horizontal",
  "children": [
    {"id": "sidebar", "type": "drawer", "flex_ratio": 3, "label": "Navigation"},
    {"id": "main", "type": "container", "flex_ratio": 9, "label": "Dashboard Widgets"}
  ]
}
```

### Stage 2 (Validation)
- [x] No duplicate IDs
- [x] Flex ratios in valid range (3, 9)
- [x] No cycles detected

### Stage 3 Output (D2 DSL)
```d2
root: container {
  direction: right

  sidebar: Navigation {
    width: 25%
  }

  main: Dashboard Widgets {
    width: 75%
  }
}
```

## Key Design Principles

1. **Schema Ownership**: The Python API owns the syntax. The LLM is a consumer that fills templates, not an inventor of structure.

2. **Constrained Generation**: By limiting the LLM's output space to known-renderable entities, we eliminate entire classes of hallucination.

3. **Human-in-the-Loop**: Visual previews enable verification before implementation, catching structural issues early.

4. **Provider Abstraction**: The IR is target-agnostic. New rendering backends (Mermaid, Excalidraw, etc.) can be added without changing the core model.

5. **Enum-Based Type Safety**: Strong typing via enums ensures compile-time verification and IDE support.

## References

- Chen et al. (2025): Structured interfaces improve LLM UI generation by ~30%
- UIFormer (Dec 2025): DSL constraints reduce token consumption by ~50%
- Rico Dataset: http://interactionmining.org/rico
- D2 Lang: https://d2lang.com/
- Kroki: https://kroki.io/
