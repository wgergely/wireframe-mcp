# Corpus Data Normalization Research

## Executive Summary

The corpus module aggregates UI data from multiple sources with fundamentally different data structures. This document analyzes each source format and proposes normalization strategies for creating a unified RAG-vectorizable schema.

## Provider Data Format Analysis

### 1. Rico / Enrico (Mobile App UI)

**Source**: Android view hierarchy dumps
**Format**: JSON tree + PNG/JPG screenshots
**Status**: ✅ Implemented and working

**Data Structure**:
```json
{
  "class": "android.widget.Button",
  "bounds": [x1, y1, x2, y2],
  "text": "Submit",
  "resource-id": "com.app:id/submit_btn",
  "clickable": true,
  "componentLabel": "Text Button",
  "children": [...]
}
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `class` | string | Android widget class |
| `bounds` | [int, int, int, int] | Bounding box coordinates |
| `text` | string | Text content |
| `componentLabel` | string | Semantic UI type (17-22 categories) |
| `children` | array | Nested child elements |

**Semantic Labels** (Rico: 17, Enrico: 22):
- Containers: Card, Drawer, Modal, Web View
- Navigation: Toolbar, Bottom Navigation, Multi-Tab, Pager Indicator
- Content: Text, Image, Icon, Background Image, Advertisement, Map View, Video
- Controls: Button, Text Button, Input, Checkbox, Radio Button, On/Off Switch, Slider

**Normalization**: Direct mapping - already tree-structured with semantic labels.

---

### 2. WebSight (Synthetic Web UI)

**Source**: HuggingFace M4 synthetic dataset
**Format**: HTML/CSS text + PNG screenshots
**Status**: ❌ Placeholder - needs implementation

**Data Structure**:
```python
{
  "image": PIL.Image,           # Screenshot
  "text": "<html>...</html>",   # Full HTML with embedded CSS
  "llm_generated_idea": str     # Website concept description
}
```

**Size**: 823K (v0.1) to 2M (v0.2) samples, 317 GB total

**Normalization Challenge**:
- HTML is flat text, not parsed structure
- Need to convert HTML DOM → tree hierarchy
- Web elements (div, h1, button) → map to semantic types

**Proposed Strategy**:
```python
# Parse HTML to extract DOM hierarchy
from bs4 import BeautifulSoup

def html_to_hierarchy(html: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    return dom_to_dict(soup.body)

def dom_to_dict(element) -> dict:
    return {
        "tag": element.name,
        "class": element.get("class", []),
        "text": element.get_text(strip=True)[:100],
        "bounds": None,  # Not available in HTML
        "componentLabel": map_html_to_component(element),
        "children": [dom_to_dict(c) for c in element.children if c.name]
    }
```

**Component Mapping** (HTML → Semantic):
| HTML Element | Semantic Type |
|--------------|---------------|
| `<button>`, `<input type="submit">` | Button |
| `<input type="text">`, `<textarea>` | Input |
| `<img>` | Image |
| `<h1>`-`<h6>`, `<p>`, `<span>` | Text |
| `<nav>` | Toolbar / Bottom Navigation |
| `<div class="card">` | Card |
| `<a>` | Text Button |

---

### 3. EGFE (Figma/Sketch Design Prototypes)

**Source**: Zenodo - Alibaba design prototypes
**Format**: JSON properties + PNG screenshots + asset images
**Status**: ❌ Placeholder - needs implementation

**Data Structure** (per sample):
```
xxx.png        - High-fidelity UI screenshot
xxx-assets.png - Stacked element images
xxx.json       - UI element properties
```

**Size**: 300 samples, 30.8 MB (partial release)

**JSON Structure** (inferred from similar Figma datasets):
```json
{
  "id": "node_123",
  "name": "Submit Button",
  "type": "RECTANGLE",
  "absoluteBoundingBox": {"x": 10, "y": 20, "width": 100, "height": 40},
  "fills": [...],
  "children": [...]
}
```

**Normalization Challenge**:
- Figma uses design-oriented types (FRAME, RECTANGLE, TEXT)
- Need to infer semantic UI types from context
- Design hierarchies may not match runtime UI structure

**Proposed Strategy**:
```python
FIGMA_TO_SEMANTIC = {
    "RECTANGLE": "container",  # Context-dependent
    "TEXT": "text",
    "FRAME": "container",
    "COMPONENT": "card",  # Usually reusable components
    "INSTANCE": "card",
}

def figma_to_hierarchy(figma_node: dict) -> dict:
    return {
        "figma_type": figma_node["type"],
        "bounds": bbox_to_bounds(figma_node.get("absoluteBoundingBox")),
        "text": figma_node.get("characters", ""),
        "componentLabel": infer_component_type(figma_node),
        "children": [figma_to_hierarchy(c) for c in figma_node.get("children", [])]
    }
```

---

### 4. ShowUI Desktop (Desktop Application UI)

**Source**: HuggingFace / OmniAct augmented dataset
**Format**: Detection annotations + keypoints + screenshots
**Status**: ❌ Placeholder - needs implementation

**Data Structure**:
```python
{
  "image": PIL.Image,
  "instruction": "Select the file menu button",
  "action_detections": {
    "detections": [{
      "label": "action",
      "bounding_box": [x, y, w, h]  # Normalized 0-1
    }]
  },
  "action_keypoints": {
    "keypoints": [{
      "label": "action",
      "points": [[x, y]]
    }]
  },
  "query_type": {"label": "appearance"},
  "interfaces": {"label": "audible"}
}
```

**Size**: 7,496 samples from ~100 screenshots

**Normalization Challenge**:
- **Flat structure**: Detections are not hierarchical
- **No tree relationships**: Each detection is independent
- **Action-oriented**: Designed for grounding, not structure

**Proposed Strategy**:
```python
def showui_to_hierarchy(sample: dict) -> dict:
    """Convert flat detections to pseudo-hierarchy."""
    # Group detections as children of a root "screen" node
    detections = sample["action_detections"]["detections"]

    return {
        "tag": "screen",
        "bounds": [0, 0, 1, 1],  # Normalized full screen
        "componentLabel": "container",
        "children": [
            {
                "tag": "element",
                "bounds": denormalize_bbox(d["bounding_box"]),
                "text": sample["instruction"],
                "componentLabel": infer_from_query(sample),
                "query_type": sample["query_type"]["label"],
            }
            for d in detections
        ]
    }
```

**Limitation**: ShowUI lacks hierarchical structure - conversions produce flat trees.

---

## Unified Normalization Schema

### StandardizedData Model (Current)

```python
class StandardizedData(BaseModel):
    id: str                      # Unique identifier
    source: str                  # Provider name
    dataset: str                 # Dataset variant
    hierarchy: dict              # Raw hierarchical data
    layout: LayoutNode | None    # Semantic MID representation
    metadata: dict               # Additional metadata
    screenshot_path: Path | None # Path to screenshot
```

### Proposed Enhanced Schema

```python
class StandardizedData(BaseModel):
    id: str
    source: str
    dataset: str

    # Raw data (format varies by source)
    hierarchy: dict              # Tree structure
    raw_format: str              # "android_view", "html_dom", "figma", "detections"

    # Normalized representation
    layout: LayoutNode | None    # Semantic MID (populated by normalizer)

    # Common fields extracted
    bounds: list[int] | None     # Root element bounds
    text_content: list[str]      # All text in UI
    component_counts: dict       # {"button": 3, "text": 10, ...}

    # Metadata
    metadata: dict
    screenshot_path: Path | None

    # RAG-specific fields
    embedding_text: str | None   # Text representation for embedding
```

### Normalization Pipeline

```
Raw Data → Format-Specific Parser → Unified Hierarchy → LayoutNode Converter → RAG Embedding
```

1. **Format-Specific Parsers**:
   - `rico_parser.py` - Android view hierarchy
   - `html_parser.py` - HTML DOM
   - `figma_parser.py` - Figma JSON
   - `detection_parser.py` - Flat detection format

2. **Unified Hierarchy Schema**:
   ```python
   {
       "type": str,           # Semantic component type
       "bounds": [int, int, int, int],
       "text": str | None,
       "attributes": dict,    # Source-specific attributes
       "children": list
   }
   ```

3. **LayoutNode Converter**:
   ```python
   def hierarchy_to_layout(node: dict) -> LayoutNode:
       return LayoutNode(
           id=generate_id(),
           type=ComponentType(node["type"]),
           label=node.get("text"),
           flex_ratio=calculate_flex(node["bounds"]),
           children=[hierarchy_to_layout(c) for c in node.get("children", [])],
           orientation=infer_orientation(node)
       )
   ```

---

## Implementation Recommendations

### Phase 1: Complete Working Providers
1. ✅ Rico - Working
2. ✅ Enrico - Working (fixed)
3. Implement hierarchy → LayoutNode conversion for Rico/Enrico

### Phase 2: Web UI Support
4. Implement WebSight with HTML DOM parsing
5. Define HTML → ComponentType mapping
6. Handle missing bounds (use relative positioning)

### Phase 3: Design Tool Support
7. Research EGFE JSON schema (need dataset access)
8. Implement Figma node → hierarchy conversion
9. Handle design-vs-runtime structure differences

### Phase 4: Detection-Based Datasets
10. Implement ShowUI with flat-to-tree conversion
11. Consider alternative representation for non-hierarchical data
12. Document limitations of converted data

### Phase 5: RAG Vectorization
13. Implement `embedding_text` generation
14. Create vector embedding pipeline
15. Build retrieval interface

---

## Open Questions

1. **Bounds normalization**: Should all bounds be normalized to 0-1 or use absolute pixels?
2. **Text extraction depth**: How deep to recurse for text content aggregation?
3. **Missing hierarchy**: How to handle ShowUI's flat structure in tree-based operations?
4. **HTML parsing depth**: Should we parse full DOM or just body content?
5. **Component type coverage**: Do we need to extend ComponentType enum for web/desktop?

---

## References

- [Rico Dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/)
- [Enrico Dataset](https://github.com/luileito/enrico)
- [WebSight Dataset](https://huggingface.co/datasets/HuggingFaceM4/WebSight)
- [EGFE Dataset](https://zenodo.org/records/8004165)
- [ShowUI Desktop](https://huggingface.co/datasets/Voxel51/ShowUI_desktop)
