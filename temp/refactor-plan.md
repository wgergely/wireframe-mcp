# MCP Tools Refactoring Plan

> Based on developer workflow analysis - simplify and clarify tool boundaries

---

## Summary of Changes

### Rename/Rebrand
| Current | New | Reason |
|---------|-----|--------|
| `text_tree` output | `draft` | Clearer purpose - quick review artifact |
| `render_layout` | `preview_layout` | Intent-based naming |

### Remove from Public API
| Tool | Reason |
|------|--------|
| `transpile_layout` | Internal implementation detail |

### Add Style Abstraction
| Current | New |
|---------|-----|
| `provider: "d2" \| "plantuml"` | `style: "wireframe" \| "sketch" \| "minimal"` |

---

## Detailed Changes

### 1. `generate_layout` - Rename output field

**Before**:
```python
{
    "layout": {...},
    "text_tree": "...",  # Confusing name
    "stats": {...}
}
```

**After**:
```python
{
    "layout": {...},
    "draft": "...",      # Clear: this is your quick draft to review
    "stats": {...}
}
```

**Files to modify**:
- `src/mcp/server.py` - Update docstring
- `src/mcp/tools/generate.py` - Rename output key
- `src/mcp/test.py` - Update tests

---

### 2. `render_layout` → `preview_layout` - Rename + Abstract Provider

**Before**:
```python
render_layout(
    layout: dict,
    format: str = "png",
    provider: str = "plantuml",  # Implementation detail exposed
)
```

**After**:
```python
preview_layout(
    layout: dict,
    style: str = "wireframe",    # Intent-based
    format: str = "png",
)
```

**Style → Provider mapping** (internal):
```python
STYLE_PROVIDERS = {
    "wireframe": ("plantuml", {}),           # Best for UI mockups
    "sketch": ("d2", {"sketch": True}),      # Hand-drawn feel
    "minimal": ("d2", {}),                   # Simple boxes
}
```

**Files to modify**:
- `src/mcp/server.py` - Rename function, update signature
- `src/mcp/tools/render.py` - Rename to `preview.py`, add style mapping
- `src/mcp/tools/__init__.py` - Update exports

---

### 3. Remove `transpile_layout` from Public API

**Action**: Keep internal function but remove `@mcp.tool` decorator

The transpilation is used internally by `preview_layout`. It should not be exposed.

**Files to modify**:
- `src/mcp/server.py` - Remove `transpile_layout` tool registration
- `src/mcp/tools/__init__.py` - Remove from exports
- `src/mcp/tools/transpile.py` - Keep as internal helper (rename to `_transpile.py` or move to `lib.py`)

---

### 4. Update `get_server_info` tool list

**Before**:
```python
"tools": [
    "ping",
    "get_server_info",
    "generate_layout",
    "validate_layout",
    "transpile_layout",  # Remove
    "render_layout",     # Rename
    "search_layouts",
]
```

**After**:
```python
"tools": [
    "ping",
    "get_server_info",
    "generate_layout",
    "validate_layout",
    "preview_layout",    # Renamed
    "search_layouts",
]
```

---

## Implementation Order

### Step 1: Rename `text_tree` → `draft`
- [ ] Update `generate.py` output key
- [ ] Update `server.py` docstring
- [ ] Update tests

### Step 2: Rename `render_layout` → `preview_layout`
- [ ] Rename `render.py` → `preview.py`
- [ ] Update function name and signature
- [ ] Add style → provider mapping
- [ ] Update `__init__.py` exports
- [ ] Update `server.py` registration

### Step 3: Internalize `transpile_layout`
- [ ] Remove `@mcp.tool` decorator from server.py
- [ ] Keep internal function for `preview_layout` to use
- [ ] Remove from `__init__.py` exports
- [ ] Update `get_server_info` tool list

### Step 4: Update Documentation
- [ ] Update DEVELOPER_GUIDE.md if needed
- [ ] Update task.md progress

### Step 5: Update Tests
- [ ] Rename test functions
- [ ] Update expected outputs
- [ ] Add style parameter tests

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/mcp/server.py` | Rename tools, remove transpile, update docstrings |
| `src/mcp/tools/generate.py` | `text_tree` → `draft` |
| `src/mcp/tools/render.py` | Rename to `preview.py`, add style mapping |
| `src/mcp/tools/transpile.py` | Make internal (remove from public API) |
| `src/mcp/tools/__init__.py` | Update exports |
| `src/mcp/test.py` | Update test names and expectations |
| `src/mcp/DEVELOPER_GUIDE.md` | Already correct (written with new names) |

---

## Verification Checklist

After implementation:

- [ ] `python . mcp info` shows correct tool list (no transpile_layout)
- [ ] `generate_layout` returns `draft` (not `text_tree`)
- [ ] `preview_layout` works with `style="wireframe"`
- [ ] `preview_layout` works with `style="sketch"`
- [ ] Old names fail gracefully (removed)
- [ ] Tests pass
- [ ] DEVELOPER_GUIDE.md matches implementation

---

## Commit Plan

1. **Commit 1**: `refactor(mcp): rename text_tree to draft in generate_layout`
2. **Commit 2**: `refactor(mcp): rename render_layout to preview_layout with style abstraction`
3. **Commit 3**: `refactor(mcp): internalize transpile_layout, remove from public API`
4. **Commit 4**: `docs(mcp): add DEVELOPER_GUIDE.md`
