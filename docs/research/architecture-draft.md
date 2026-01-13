# Layout Broker - Mission Statement & Architecture

## Mission

Enable users to describe UI layouts in natural language and receive verified structural blueprints before implementation. The system acts as a **translation and validation layer** between human intent and machine-renderable layout syntax.

---

## The Problem

When a user says *"Make a dashboard with a sidebar and floating search"*, the typical LLM response jumps directly to implementation code. This creates:

1. **The Black Box Effect**: No visibility into what the LLM "imagines"
2. **Spatial Blindness**: LLMs lack inherent understanding of containment, overlap, docking
3. **Wasted Iterations**: Errors discovered only after code is written

---

## The Solution: A Three-Layer Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1: NATURAL LANGUAGE                   │
│                                                                 │
│  User Input: "A dashboard with sidebar on left, header bar,    │
│               and a floating search overlay"                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 2: LLM TRANSLATOR                     │
│                                                                 │
│  • Consumes natural language                                    │
│  • Produces CONSTRAINED syntax (not free-form code)             │
│  • Guided by schema/grammar provided by Layer 3                 │
│                                                                 │
│  Key Insight: The LLM does NOT invent the syntax.               │
│  It fills in a structured template defined by Layer 3.          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 3: PYTHON API BACKEND                    │
│                                                                 │
│  RESPONSIBILITIES:                                               │
│  1. Define syntax grammar (what is valid Layout-IR)             │
│  2. Validate LLM output (reject invalid structures)             │
│  3. Translate to provider-specific DSL (D2, PlantUML)           │
│  4. Render to visual preview via Kroki                          │
│                                                                 │
│  This layer OWNS the schema. The LLM is a consumer.             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 4: UI GENERATORS                       │
│                                                                 │
│  • Kroki renders DSL → PNG/SVG for user verification            │
│  • Future: Approved layouts → implementation code               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer Responsibilities

### Layer 1: Natural Language (User)
- **Input**: Free-form description of desired UI
- **Output**: Text passed to Layer 2
- **Owner**: End user

### Layer 2: LLM Translator (External)
- **Input**: Natural language + syntax schema from Layer 3
- **Output**: Structured Layout-IR (JSON or DSL subset)
- **Owner**: External LLM (Claude, GPT, Gemini)
- **Key Constraint**: LLM output MUST conform to schema provided by Layer 3

### Layer 3: Python API Backend (This Project)
- **Input**: Layout-IR from Layer 2
- **Output**: Validated DSL code + rendered preview
- **Owner**: This codebase
- **Responsibilities**:
  - Define `LayoutAST` schema (what nodes, constraints, directions are valid)
  - Provide schema to Layer 2 (via prompts, JSON Schema, or MCP resources)
  - Validate incoming Layout-IR
  - Generate provider-specific DSL (D2, PlantUML)
  - Call Kroki to render visual preview

### Layer 4: UI Generators (Rendering)
- **Input**: DSL code from Layer 3
- **Output**: Visual image (PNG/SVG)
- **Owner**: Kroki (external service)

---

## Intended Flow

```
1. User describes layout in natural language
           │
           ▼
2. LLM receives NL + LayoutAST schema
   LLM produces structured Layout-IR (not code)
           │
           ▼
3. Python API validates Layout-IR
   Python API translates to D2/PlantUML
   Python API calls Kroki
           │
           ▼
4. User sees visual preview
   User approves OR requests changes
           │
           ▼
5. Loop back to step 1 if changes needed
```

---

## Key Insight: Schema Ownership

The Python API **owns the syntax**. The LLM is a **consumer** of that syntax.

This means:
- Layer 3 defines what `LayoutNode`, `LayoutAST`, `constraints` look like
- Layer 3 exports a JSON Schema or grammar for Layer 2 to use
- Layer 2 (LLM) fills in the schema, it does NOT invent the structure
- If Layer 2 produces invalid output, Layer 3 rejects it with clear errors

This prevents LLM hallucination of impossible layouts.

---

## Open Questions

1. **Schema Format**: Should we provide JSON Schema, Pydantic models, or a custom grammar to the LLM?
2. **LLM Integration**: Is Layer 2 inside or outside the MCP server?
3. **Iteration UX**: How does the user request changes? Edit NL, edit preview, or edit raw syntax?
