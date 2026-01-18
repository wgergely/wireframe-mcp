# CLI to MCP Tool Mapping

> **Purpose**: Reference for converting existing CLI commands to MCP tools
> **Last Updated**: 2026-01-18

---

## Overview

This document maps existing CLI functionality to proposed MCP tools, identifying the code that needs to be wrapped.

---

## Tool Mappings

### 1. generate_layout

**CLI Command**: `python . generate layout "query"`

**Source Code**: `__main__.py:35-122` → `cmd_generate_layout()`

**Key Dependencies**:
- `src.llm.LayoutGenerator`
- `src.llm.GeneratorConfig`
- `src.llm.create_llm_backend`
- `src.vector.VectorStore` (optional, for RAG)
- `src.output.OutputGenerator`

**MCP Tool Signature**:
```python
@mcp.tool
def generate_layout(
    query: str,
    model: str | None = None,
    temperature: float = 0.7,
    target: str = "d2",
    include_rag: bool = True,
) -> dict:
    """Generate UI layout from natural language description.

    Args:
        query: Natural language description of the layout
        model: LLM model name (e.g., 'gpt-4.1-mini', 'claude-sonnet-4-5')
        temperature: Generation temperature (0.0-2.0)
        target: Target transpiler ('d2' or 'plantuml')
        include_rag: Include similar layouts from vector store

    Returns:
        Dictionary with:
        - layout: Generated LayoutNode as JSON
        - dsl: Transpiled DSL code
        - stats: Generation statistics
    """
```

**Implementation Notes**:
- Wrap `LayoutGenerator.generate()` call
- Include DSL transpilation in response
- Return JSON-serializable output

---

### 2. search_layouts

**CLI Command**: `python . search "query" -k 5`

**Source Code**: `__main__.py:593-626` → `cmd_search_index()`

**Key Dependencies**:
- `src.vector.VectorStore`
- `src.config.get_index_dir`

**MCP Tool Signature**:
```python
@mcp.tool
def search_layouts(
    query: str,
    k: int = 5,
) -> list[dict]:
    """Search for similar layouts in the vector database.

    Args:
        query: Search query describing desired layout
        k: Number of results to return (1-20)

    Returns:
        List of dictionaries with:
        - score: Similarity score (0-1)
        - text: Layout text representation
        - metadata: Source information
    """
```

**Implementation Notes**:
- Load index from default location
- Cap k at reasonable maximum (20)
- Format results for readability

---

### 3. render_layout

**CLI Command**: `python . demo {layout} --render -f png`

**Source Code**: `__main__.py:1172-1330` → `cmd_demo_render()`

**Key Dependencies**:
- `src.render.RenderClient`
- `src.render.RenderConfig`
- `src.render.OutputFormat`
- `src.providers.get_provider`

**MCP Tool Signature**:
```python
@mcp.tool
def render_layout(
    layout: dict | str,
    provider: str = "plantuml",
    format: str = "png",
) -> dict:
    """Render a layout to an image.

    Args:
        layout: Layout as JSON dict or DSL string
        provider: Transpiler to use ('d2' or 'plantuml')
        format: Output format ('png' or 'svg')

    Returns:
        Dictionary with:
        - data: Base64-encoded image data
        - format: Output format used
        - size_bytes: Image size in bytes
    """
```

**Implementation Notes**:
- Accept both JSON layouts and DSL strings
- Return base64-encoded image data
- Require Kroki service to be running

---

### 4. validate_layout

**CLI Command**: (Internal validation in generate flow)

**Source Code**: `src/mid/lib.py` → `validate_layout()`

**Key Dependencies**:
- `src.mid.LayoutNode`
- `src.mid.validate_layout`
- `src.mid.ValidationError`

**MCP Tool Signature**:
```python
@mcp.tool
def validate_layout(
    layout: dict,
) -> dict:
    """Validate a layout structure.

    Args:
        layout: Layout JSON to validate

    Returns:
        Dictionary with:
        - valid: Boolean indicating if layout is valid
        - errors: List of validation errors (if any)
        - node_count: Number of nodes in layout
    """
```

**Implementation Notes**:
- Parse JSON to LayoutNode
- Run validation checks
- Return structured error information

---

### 5. transpile_layout

**CLI Command**: (Internal transpilation in generate flow)

**Source Code**: `src/providers/lib.py` → `get_provider().transpile()`

**Key Dependencies**:
- `src.providers.get_provider`
- `src.providers.d2.D2Provider`
- `src.providers.plantuml.PlantUMLProvider`

**MCP Tool Signature**:
```python
@mcp.tool
def transpile_layout(
    layout: dict,
    provider: str = "d2",
) -> dict:
    """Transpile layout JSON to DSL code.

    Args:
        layout: Layout JSON to transpile
        provider: Target DSL ('d2' or 'plantuml')

    Returns:
        Dictionary with:
        - dsl: Generated DSL code
        - provider: Provider used
        - line_count: Number of lines generated
    """
```

**Implementation Notes**:
- Parse layout to LayoutNode
- Use appropriate provider
- Return DSL string

---

## Resource Mappings

### 1. schema://components

**Source**: `src/schema/lib.py` → `export_llm_schema()`

**Returns**: JSON schema of all component types with metadata

---

### 2. schema://layout

**Source**: `src/mid/lib.py` → `LayoutNode.model_json_schema()`

**Returns**: JSON schema for LayoutNode model

---

### 3. config://models

**Source**: `src/llm/backend/models.py` → `LLMModel.list_all()`

**Returns**: List of available LLM models

---

### 4. config://providers

**Source**: `src/corpus/api/lib.py` → `CorpusManager.list_providers()`

**Returns**: List of available corpus providers

---

## Existing Code to Reuse

### From `__main__.py`

| Function | Purpose | Lines |
|----------|---------|-------|
| `cmd_generate_layout` | Layout generation | 35-122 |
| `cmd_search_index` | Vector search | 593-626 |
| `cmd_demo_render` | Render pipeline | 1172-1330 |
| `_check_llm_availability` | LLM provider check | 1387-1445 |
| `_check_kroki_health` | Kroki health check | 1448-1457 |

### From `src/llm/generator/lib.py`

| Class | Purpose | Lines |
|-------|---------|-------|
| `LayoutGenerator` | Core generation | 84-365 |
| `GeneratorConfig` | Configuration | 24-42 |
| `GenerationOutput` | Output model | 63-78 |

### From `src/providers/`

| Module | Purpose |
|--------|---------|
| `src/providers/lib.py` | Provider factory |
| `src/providers/d2/lib.py` | D2 transpiler |
| `src/providers/plantuml/lib.py` | PlantUML transpiler |

### From `src/vector/`

| Module | Purpose |
|--------|---------|
| `src/vector/lib.py` | VectorStore class |
| `src/vector/backend/` | Embedding backends |

---

## Error Handling Mapping

| CLI Error | MCP Handling |
|-----------|--------------|
| Model not found | Return error with available models |
| Index not found | Return error with setup instructions |
| Kroki unavailable | Return error with service status |
| Generation failed | Return error with retry suggestion |
| Validation failed | Return validation errors list |

---

## Configuration Mapping

| CLI Config | MCP Equivalent |
|------------|----------------|
| `--model` | Tool parameter |
| `--temperature` | Tool parameter |
| `--index` | Server config (env var) |
| `--output` | N/A (return data directly) |
| `--format` | Tool parameter |

---

## Implementation Checklist

- [ ] Create `src/mcp/tools/generate.py` wrapping `LayoutGenerator`
- [ ] Create `src/mcp/tools/search.py` wrapping `VectorStore.search`
- [ ] Create `src/mcp/tools/render.py` wrapping `RenderClient`
- [ ] Create `src/mcp/tools/validate.py` wrapping MID validation
- [ ] Create `src/mcp/tools/transpile.py` wrapping providers
- [ ] Create `src/mcp/resources.py` for schema/config resources
- [ ] Wire all to `src/mcp/server.py`
