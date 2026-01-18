# MCP Tool Workflow Design

> **Purpose**: Define the expected developer workflow when using MCP tools
> **Last Updated**: 2026-01-18

---

## Key Insight

`generate_layout` should return **structured data for review**, NOT rendered images.

The workflow is:
1. **Generate** → Get JSON structure + text tree for quick inspection
2. **Review** → Developer inspects the text tree representation
3. **Iterate** → Regenerate with refined prompt if not satisfied
4. **Preview** → Render to image once structure looks good
5. **Export** → Get DSL code for final use

---

## Data Flow

```
Natural Language
       ↓
   [generate_layout]
       ↓
   LayoutNode (JSON) + Text Tree
       ↓
   Developer Reviews
       ↓
   ┌─── Satisfied? ───┐
   │                  │
   No                Yes
   │                  │
   ↓                  ↓
Refine query    [render_layout]
& regenerate          ↓
                 PNG/SVG Preview
                      ↓
               [transpile_layout]
                      ↓
                 D2/PlantUML DSL
```

---

## Tool Outputs

### generate_layout

**Purpose**: Create structured layout from natural language

**Input**:
```python
{
    "query": "login page with email, password, and forgot password link",
    "model": "gpt-4.1-mini",        # optional
    "temperature": 0.7,              # optional
    "provider": "d2",                # optional, for text_tree hints
}
```

**Output**:
```python
{
    "layout": {                      # Full JSON structure
        "id": "root",
        "type": "container",
        "orientation": "vertical",
        "children": [...]
    },
    "text_tree": """                 # Quick human inspection
        Login Page [container, vertical]
        └── Form [card]
            ├── Email [input]
            ├── Password [input]
            └── Forgot Password [button]
    """,
    "stats": {
        "attempts": 1,
        "tokens": 150,
        "model": "gpt-4.1-mini"
    }
}
```

**Why this design**:
- JSON is the canonical representation (programmable)
- Text tree enables quick review without rendering
- Stats help track costs and debug issues
- NO image - that's a separate explicit step

---

### render_layout

**Purpose**: Render layout to visual image

**Input**:
```python
{
    "layout": {...},                 # JSON from generate_layout
    "format": "png",                 # or "svg"
    "provider": "plantuml"           # or "d2"
}
```

**Output**:
```python
{
    "image_data": "base64...",       # Encoded image
    "format": "png",
    "size_bytes": 12345,
    "provider": "plantuml"
}
```

**Why separate**:
- Rendering requires Kroki service (may not be available)
- Rendering is slower than text generation
- Developer may not need visual preview every time
- Explicit step gives control over when to incur cost

---

### transpile_layout

**Purpose**: Convert layout JSON to DSL code

**Input**:
```python
{
    "layout": {...},
    "provider": "d2"                 # or "plantuml"
}
```

**Output**:
```python
{
    "dsl_code": "root: container {\n  ...\n}",
    "provider": "d2",
    "line_count": 25
}
```

**Why separate**:
- Developer may want to see/edit DSL code
- Different providers have different aesthetics
- Can export DSL without rendering

---

### validate_layout

**Purpose**: Check layout structure for errors

**Input**:
```python
{
    "layout": {...}
}
```

**Output**:
```python
{
    "valid": true,
    "errors": [],
    "warnings": [
        {"node_id": "btn1", "message": "Button has no label"}
    ],
    "stats": {
        "node_count": 12,
        "max_depth": 4,
        "component_types": ["container", "input", "button"]
    }
}
```

---

### search_layouts

**Purpose**: Find similar layouts in corpus

**Input**:
```python
{
    "query": "settings page with toggles",
    "k": 5
}
```

**Output**:
```python
{
    "results": [
        {
            "score": 0.89,
            "text": "Settings screen with...",
            "metadata": {"source": "rico", "id": "12345"}
        },
        ...
    ],
    "total_in_index": 15000
}
```

---

## Developer Workflow Examples

### Example 1: Simple Generation

```
Developer: "Create a login form"

→ call generate_layout(query="login form with email and password")

← {
    layout: {...},
    text_tree: """
        Login [container]
        └── Form [card]
            ├── Email [input]
            ├── Password [input]
            └── Submit [button]
    """
  }

Developer reviews text_tree: "Looks good, let me see it"

→ call render_layout(layout=..., format="png")

← {image_data: "base64...", format: "png"}

Developer: "Perfect, give me the D2 code"

→ call transpile_layout(layout=..., provider="d2")

← {dsl_code: "root: Login {...}"}
```

### Example 2: Iterative Refinement

```
Developer: "Dashboard with sidebar"

→ call generate_layout(query="dashboard with sidebar")

← {
    text_tree: """
        Dashboard [container, horizontal]
        ├── Sidebar [drawer, 25%]
        └── Main [container, 75%]
    """
  }

Developer: "Sidebar needs navigation items"

→ call generate_layout(query="dashboard with sidebar containing Home, Settings, Profile links, and main content area with stats cards")

← {
    text_tree: """
        Dashboard [container, horizontal]
        ├── Sidebar [drawer, 25%]
        │   ├── Home [button]
        │   ├── Settings [button]
        │   └── Profile [button]
        └── Main [container, 75%]
            └── Stats [container, horizontal]
                ├── Card 1 [card]
                ├── Card 2 [card]
                └── Card 3 [card]
    """
  }

Developer: "Better! Show me a preview"

→ call render_layout(layout=..., format="png", provider="plantuml")

← {image_data: "..."}
```

### Example 3: Using Search for Inspiration

```
Developer: "What do settings pages look like?"

→ call search_layouts(query="settings page with user preferences", k=3)

← {
    results: [
        {score: 0.92, text: "Settings with profile, notifications, privacy sections..."},
        {score: 0.88, text: "Preferences panel with toggles and sliders..."},
        {score: 0.85, text: "Account settings with sections..."}
    ]
  }

Developer: "Generate something like the first one"

→ call generate_layout(query="settings page with profile section, notifications toggles, and privacy settings, organized in tabs")

← {...}
```

---

## Variations (Future Enhancement)

For generating multiple options at once, add a parameter:

```python
generate_layout(
    query="login form",
    variations=3,          # Generate 3 different approaches
)

# Returns:
{
    "layouts": [
        {"layout": {...}, "text_tree": "..."},
        {"layout": {...}, "text_tree": "..."},
        {"layout": {...}, "text_tree": "..."},
    ],
    "stats": {...}
}
```

This would be useful for:
- Exploring design alternatives
- A/B testing options
- Getting creative inspiration

For Phase 2, we'll start with single generation. Variations can be added in Phase 4 (agentic mode) where session state allows comparing and selecting.

---

## Phase 4 Preview: Agentic Refinement

With session state (Phase 4), the workflow becomes:

```
→ call start_session(query="login form")
← {session_id: "abc", layout: {...}, text_tree: "..."}

→ call refine(session_id="abc", feedback="center the form")
← {layout: {...}, text_tree: "..."}  # Updated

→ call refine(session_id="abc", feedback="add social login buttons")
← {layout: {...}, text_tree: "..."}  # Updated again

→ call get_variations(session_id="abc", count=3)
← {variations: [...]}

→ call undo(session_id="abc")
← {layout: {...}}  # Reverted to previous

→ call end_session(session_id="abc")
← {final_layout: {...}, history_count: 3}
```

This enables true conversational design refinement.

---

## Summary

| Tool | Returns | When to Use |
|------|---------|-------------|
| `generate_layout` | JSON + text_tree | Start here, iterate on query |
| `validate_layout` | Errors/warnings | Check before rendering |
| `render_layout` | PNG/SVG image | Visual preview |
| `transpile_layout` | DSL code | Export for use |
| `search_layouts` | Similar examples | Find inspiration |

The key principle: **Structured data first, visual render on demand**.
