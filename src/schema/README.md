# Schema Module

The Schema module is the **authoritative source of truth** for all UI component definitions in the wireframe system. It provides rich component metadata, structural constraints, and LLM-optimized schema exports.

## Key Exports

| Export | Description |
|--------|-------------|
| `ComponentType` | Rico-based 26-type UI taxonomy enum |
| `ComponentMeta` | Rich metadata (descriptions, aliases, constraints) |
| `ComponentCategory` | High-level groupings (Container, Navigation, Content, Control) |
| `Orientation` | Layout flow direction (Horizontal, Vertical, Overlay) |
| `export_json_schema()` | Generate Pydantic JSON Schema for LLM integration |
| `export_llm_schema()` | LLM-optimized schema with examples and descriptions |

## Component Taxonomy

The system uses a **26-category taxonomy** derived from the Rico dataset's semantic annotations. This provides a standardized vocabulary for UI components that LLMs can reliably target.

### Current Implementation (26 types)

| Category | Types |
|----------|-------|
| **Containers** (4) | `container`, `card`, `modal`, `web_view` |
| **Navigation** (7) | `toolbar`, `navbar`, `bottom_nav`, `drawer`, `tab_bar`, `multi_tab`, `pager_indicator` |
| **Content** (5) | `text`, `image`, `list_item`, `icon`, `advertisement` |
| **Controls** (10) | `button`, `text_button`, `input`, `checkbox`, `radio_button`, `switch`, `slider`, `spinner`, `date_picker`, `number_stepper` |

### Layout Properties

The schema defines comprehensive layout properties mapping to CSS Flexbox/Grid concepts:

| Property | Enum | Values |
|----------|------|--------|
| `orientation` | `Orientation` | horizontal, vertical, overlay |
| `alignment` | `Alignment` | start, center, end, stretch |
| `justify` | `Justify` | start, center, end, between, around, evenly |
| `wrap` | `Wrap` | none, wrap |
| `text_size` | `TextSize` | title, heading, body, caption |
| `text_weight` | `TextWeight` | light, normal, bold |
| `semantic_color` | `SemanticColor` | default, primary, secondary, success, warning, danger, info |

## Research: UI Element Taxonomy

This section documents research into comprehensive UI vocabularies from authoritative sources.

### Authoritative Sources

| Source | Elements | Authority |
|--------|----------|-----------|
| [NN/Group UI Glossary](https://www.nngroup.com/articles/ui-elements-glossary/) | 61 | Highest - premier UX research org |
| [Component Gallery](https://component.gallery/components/) | 73 | High - cross-design-system aggregate |
| [Atlassian Design System](https://atlassian.design/components/) | 60+ | High - production-grade system |
| Rico Dataset | 24 | Medium - academic research |

### Gap Analysis

Elements present in authoritative sources but missing from current implementation:

#### High-Value Missing Elements

| Category | Element | Found In |
|----------|---------|----------|
| Containers | `accordion`, `carousel`, `table`, `form`, `sidebar` | NN/G, Component Gallery |
| Navigation | `breadcrumb`, `link`, `pagination`, `menu`, `segmented_control` | All sources |
| Content | `badge`, `avatar`, `tag`, `tooltip`, `progress_bar`, `skeleton`, `empty_state` | NN/G, Atlassian |
| Controls | `dropdown`, `combobox`, `file_upload`, `textarea`, `search_input` | All sources |
| Feedback | `toast`, `alert`, `dialog`, `popover`, `inline_message` | New category needed |

### Recommended 50-Element Taxonomy

Future expansion target based on cross-referencing authoritative sources:

```
Containers (9):    container, card, modal, web_view, accordion, carousel, table, form, sidebar
Navigation (12):   toolbar, navbar, bottom_nav, drawer, tab_bar, multi_tab, pager_indicator,
                   breadcrumb, link, pagination, menu, segmented_control
Content (12):      text, image, list_item, icon, advertisement, badge, avatar, tag, tooltip,
                   progress_bar, skeleton, empty_state
Controls (12):     button, text_button, input, checkbox, radio_button, switch, slider, spinner,
                   date_picker, number_stepper, dropdown, combobox, file_upload, textarea, search_input
Feedback (5):      toast, alert, dialog, popover, inline_message
```

### Implementation Priority

1. Expand `ComponentType` enum to 50 elements
2. Add `ComponentCategory.FEEDBACK` for messaging elements
3. Update `COMPONENT_REGISTRY` with metadata
4. Build extraction layer to map corpus data to new taxonomy

## Usage

```python
from src.schema import (
    ComponentType,
    ComponentMeta,
    Orientation,
    export_json_schema,
    export_llm_schema,
)

# Get component metadata
meta = COMPONENT_REGISTRY[ComponentType.BUTTON]
print(meta.description)  # "Clickable action trigger"
print(meta.aliases)      # ("btn", "action")

# Export schema for LLM prompts
llm_schema = export_llm_schema()
```

## References

- [NN/Group UI Elements Glossary](https://www.nngroup.com/articles/ui-elements-glossary/)
- [Component Gallery](https://component.gallery/components/)
- [Rico Dataset](http://interactionmining.org/rico)
- [CareerFoundry UI Guide](https://careerfoundry.com/en/blog/ui-design/ui-element-glossary/)
