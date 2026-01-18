# Wireframe-MCP Developer Guide

> How to use the MCP server for AI-assisted UI layout design

---

## Overview

Wireframe-MCP helps you design UI layouts using natural language. You describe what you want, review the structure, preview it visually, and get a structured layout you can pass to code generators.

**Key Concept**: The output is a **semantic JSON layout** - not code for a specific framework. This layout can be transformed into React, Qt, SwiftUI, HTML, or any UI framework.

---

## Core Workflow

```
┌─────────────────┐
│  Natural Lang   │  "settings page with toggles..."
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ generate_layout │  Returns JSON layout + text draft
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Review Draft   │  Quick text tree - iterate if needed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ preview_layout  │  Visual wireframe image
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Approved     │  Pass JSON layout to code generator
└─────────────────┘
```

---

## Tools Reference

### generate_layout

**Purpose**: Create a UI layout from natural language description.

```python
generate_layout(
    query: str,              # What you want to build
    model: str = None,       # LLM model (optional)
    temperature: float = 0.7 # Creativity (optional)
)
```

**Returns**:
```python
{
    "layout": {...},    # Structured JSON - THE artifact
    "draft": "...",     # Text tree for quick review
    "stats": {...}      # Generation metadata
}
```

**Example**:
```
→ generate_layout(query="login form with email, password, and forgot password link")

← {
    "layout": {
        "id": "login-form",
        "type": "container",
        "children": [
            {"id": "email", "type": "input", "label": "Email"},
            {"id": "password", "type": "input", "label": "Password"},
            {"id": "forgot", "type": "link", "label": "Forgot Password?"},
            {"id": "submit", "type": "button", "label": "Sign In"}
        ]
    },
    "draft": """
        Login Form [container]
        ├── Email [input]
        ├── Password [input]
        ├── Forgot Password? [link]
        └── Sign In [button]
    """
}
```

**Usage**: Iterate on the `query` until the `draft` looks right. The `draft` is your quick feedback loop - you can scan it in seconds without waiting for image rendering.

---

### preview_layout

**Purpose**: Render a visual wireframe of your layout.

```python
preview_layout(
    layout: dict,           # JSON layout from generate_layout
    style: str = "wireframe", # Visual style
    format: str = "png"     # Image format (png/svg)
)
```

**Returns**:
```python
{
    "image_data": "base64...",  # Encoded image
    "format": "png",
    "style": "wireframe"
}
```

**Styles**:
| Style | Description | Best For |
|-------|-------------|----------|
| `wireframe` | Clean UI mockup (default) | App/web interfaces |
| `sketch` | Hand-drawn appearance | Early concepts |
| `minimal` | Simple boxes | Architecture diagrams |

**Example**:
```
→ preview_layout(layout=my_layout, style="wireframe")

← {
    "image_data": "iVBORw0KGgo...",  # Base64 PNG
    "format": "png"
}
```

**Usage**: Call this when you want visual confirmation. The draft text is usually enough for iteration - save preview for final verification.

---

### validate_layout

**Purpose**: Check a layout for structural errors.

```python
validate_layout(
    layout: dict  # JSON layout to validate
)
```

**Returns**:
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

**Usage**: Call before passing layout to code generators to catch issues early.

---

### search_layouts

**Purpose**: Find similar layouts in the corpus for inspiration.

```python
search_layouts(
    query: str,     # What you're looking for
    k: int = 5      # Number of results
)
```

**Returns**:
```python
{
    "results": [
        {"score": 0.89, "text": "Settings with toggles...", "metadata": {...}},
        {"score": 0.85, "text": "Preferences panel...", "metadata": {...}}
    ],
    "total_in_index": 15000
}
```

**Usage**: Search before generating to see existing patterns and get ideas.

---

## Complete Example: Building a Settings Page

### Step 1: Generate Initial Layout

```
Me: I need a settings page for my React app

→ generate_layout(
    query="settings page with header (back button, title),
           three sections (Account, Notifications, Privacy)
           each with toggle switches, and a save button"
  )

← {
    "layout": {
        "id": "settings",
        "type": "container",
        "orientation": "vertical",
        "children": [
            {"id": "header", "type": "toolbar", "children": [
                {"id": "back", "type": "button", "label": "←"},
                {"id": "title", "type": "text", "label": "Settings"}
            ]},
            {"id": "account", "type": "card", "label": "Account", "children": [
                {"id": "profile-toggle", "type": "toggle", "label": "Public Profile"},
                {"id": "email-toggle", "type": "toggle", "label": "Show Email"}
            ]},
            {"id": "notifications", "type": "card", "label": "Notifications", "children": [
                {"id": "push-toggle", "type": "toggle", "label": "Push"},
                {"id": "digest-toggle", "type": "toggle", "label": "Email Digest"}
            ]},
            {"id": "privacy", "type": "card", "label": "Privacy", "children": [
                {"id": "analytics-toggle", "type": "toggle", "label": "Analytics"},
                {"id": "ads-toggle", "type": "toggle", "label": "Personalized Ads"}
            ]},
            {"id": "save", "type": "button", "label": "Save Changes"}
        ]
    },
    "draft": """
        Settings [container, vertical]
        ├── Header [toolbar]
        │   ├── ← [button]
        │   └── Settings [text]
        ├── Account [card]
        │   ├── Public Profile [toggle]
        │   └── Show Email [toggle]
        ├── Notifications [card]
        │   ├── Push [toggle]
        │   └── Email Digest [toggle]
        ├── Privacy [card]
        │   ├── Analytics [toggle]
        │   └── Personalized Ads [toggle]
        └── Save Changes [button]
    """
}
```

### Step 2: Review and Iterate

Looking at the draft, I want to add an avatar to the Account section.

```
→ generate_layout(
    query="settings page with header (back button, title),
           Account section with avatar image and username at top then toggles,
           Notifications and Privacy sections with toggles,
           save button at bottom"
  )

← {draft: "...updated with avatar and username..."}
```

### Step 3: Visual Preview

```
→ preview_layout(layout=settings_layout)

← {image_data: "base64...", format: "png"}
```

I see the wireframe. Looks good!

### Step 4: Validate Before Handoff

```
→ validate_layout(layout=settings_layout)

← {valid: true, errors: [], warnings: []}
```

### Step 5: Pass to Code Generator

Now I take the `layout` JSON and pass it to my code generator:

```
"Generate a React component for this layout: {layout JSON}"
```

The code generator receives semantic structure (container, card, toggle, button) and produces actual React code.

---

## The Layout JSON: Your Handoff Artifact

The JSON layout is **framework-agnostic semantic structure**. It describes:
- **What** components exist (button, input, toggle, card)
- **How** they're organized (container, orientation, nesting)
- **What** they're labeled (text content)

It does NOT describe:
- Specific framework code (React, Qt, etc.)
- Exact styling (colors, fonts, spacing values)
- Implementation details

This makes it perfect for passing to LLMs or code generators that can target any framework.

**Example handoff prompt**:
```
Generate a React functional component for this UI layout.
Use Tailwind CSS for styling. The layout structure is:

{paste layout JSON}

Create appropriate components for each node type:
- container → div with flex
- card → Card component with shadow
- toggle → Switch component
- button → Button component
- input → Input component with label
```

---

## Tips

1. **Iterate on the draft** - It's faster than waiting for images
2. **Be specific in queries** - "login form" vs "login form with email, password, remember me checkbox, forgot password link, and social login buttons"
3. **Use search for inspiration** - See what patterns exist before generating
4. **Validate before handoff** - Catch structural issues early
5. **The layout JSON is the artifact** - That's what you pass to code generators

---

## Resources Available

Access these via MCP resources for schema information:

| Resource | Content |
|----------|---------|
| `schema://components` | All 26 component types with metadata |
| `schema://layout` | LayoutNode JSON schema |
| `config://models` | Available LLM models |
| `config://providers` | Available corpus providers |

---

## Architecture Note

Internally, the server uses D2 and PlantUML for rendering wireframes, but these are **implementation details**. You never need to see or work with DSL code. The server handles:

```
Your layout JSON
    → [internal transpilation to diagram DSL]
    → [internal rendering via Kroki]
    → PNG/SVG image back to you
```

This abstraction lets us improve rendering without changing your workflow.
