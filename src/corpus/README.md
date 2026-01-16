# Corpus Module

The Corpus module provides a **provider-based architecture** for ingesting, normalizing, and managing UI layout datasets. It transforms diverse data sources into a unified format suitable for vector embedding and RAG retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Corpus Module                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Providers  │ -> │ Normalizer  │ -> │  StandardizedData   │  │
│  │  (Rico,     │    │  (Format    │    │  (Unified schema    │  │
│  │   WebSight, │    │   specific  │    │   for vector DB)    │  │
│  │   ShowUI)   │    │   parsers)  │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Submodules

| Path | Purpose |
|------|---------|
| `corpus/provider/` | Dataset provider implementations (Rico, Enrico, WebSight, ShowUI, EGFE) |
| `corpus/normalizer/` | Format-specific parsers to unified hierarchy |
| `corpus/api/` | CorpusManager for aggregating providers |

## Usage

```python
# CLI usage
python . corpus download --provider rico --dataset semantic
python . corpus list
python . corpus process --provider rico

# Programmatic usage
from src.corpus.api import CorpusManager
from src.corpus.provider.rico import RicoProvider

manager = CorpusManager()
manager.register(RicoProvider())

for sample in manager.stream():
    print(sample.id, sample.layout)
```

---

## Dataset Catalog

### Mobile UI Datasets

| Dataset | Size | Platform | Key Features | Status |
|---------|------|----------|--------------|--------|
| **Rico** | 72k screens | Android | 26 component categories, semantic annotations | ✅ Implemented |
| **Enrico** | 1.4k screens | Android | Rico subset with design topics | ✅ Implemented |

### Web UI Datasets

| Dataset | Size | Format | Key Features | Status |
|---------|------|--------|--------------|--------|
| **WebSight** | 2M samples | HTML + PNG | Synthetic HTML/Tailwind | ✅ Implemented |
| **WebUI** | 400k pages | JSON + PNG | Full-page screenshots, accessibility trees | Planned |

### Desktop UI Datasets

| Dataset | Size | Platform | Key Features | Status |
|---------|------|----------|--------------|--------|
| **ShowUI Desktop** | 7.5k | 15 apps | 8k element annotations | ✅ Implemented |
| **ScreenSpot** | Multi-platform | macOS/Windows | Apache 2.0 license | Planned |

### Design-Focused Datasets

| Dataset | Size | Format | Key Features | Status |
|---------|------|--------|--------------|--------|
| **EGFE** | High-quality | Sketch/Figma JSON | Layered design prototypes | ✅ Implemented |

---

## Dataset Suitability Ranking

Ranked by **Availability**, **License**, and **Normalizability** for vector database training:

| Rank | Dataset | License | Normalizability | Verdict |
|:-----|:--------|:--------|:----------------|:--------|
| **1** | **Enrico** | MIT | 5/5 - Cleaned Rico subset | 🏆 Best Starter |
| **2** | **Rico (Semantic)** | CC-BY-4.0 | 5/5 - Standard JSON | ✅ Standard Corpus |
| **3** | **WebSight** | Research | 4/5 - HTML needs parsing | React/Web Focus |
| **4** | **ShowUI Desktop** | Research | 3/5 - Flat detections | Desktop Focus |
| **5** | **WebUI** | Fair use | 3/5 - Noisy HTML | Volume Play |

### Acquisition Strategy

1. **Phase 1 (MVP)**: Enrico (1.4k screens) - Small, MIT, pre-cleaned
2. **Phase 2 (Scale)**: Rico Semantic (72k screens) - CC-BY-4.0
3. **Phase 3 (Web)**: WebSight for HTML/React layouts

---

## Data Normalization

### Provider Data Formats

Each provider has a unique data structure requiring format-specific parsing:

| Provider | Source Format | Normalization Strategy |
|----------|---------------|------------------------|
| Rico/Enrico | Android View Hierarchy JSON | Direct mapping - already tree-structured |
| WebSight | HTML/CSS text | Parse DOM with BeautifulSoup → tree |
| ShowUI | Flat detection annotations | Group as children of root "screen" node |
| EGFE | Figma JSON | Infer semantic types from design primitives |

### StandardizedData Schema

```python
class StandardizedData(BaseModel):
    id: str                      # Unique identifier
    source: str                  # Provider name (e.g., "rico")
    dataset: str                 # Dataset variant (e.g., "semantic")
    hierarchy: dict              # Raw hierarchical data
    layout: LayoutNode | None    # Semantic MID representation
    metadata: dict               # Additional metadata
    screenshot_path: Path | None # Path to screenshot
```

### Normalization Pipeline

```
Raw Data → Format-Specific Parser → Unified Hierarchy → LayoutNode Converter → RAG Embedding
```

1. **Format-Specific Parsers**: Handle source quirks
2. **Unified Hierarchy**: Common tree structure
3. **LayoutNode Converter**: Map to MID schema
4. **Embedding Text**: Generate text for vector embedding

---

## Rico Dataset Details

**CLI**: `python . corpus download --provider rico --dataset semantic`

### Available Subsets

| Subset | Size | Description |
|--------|------|-------------|
| `semantic` ✅ | 150MB | Semantic annotations (72k screens) |
| `ui_screenshots` | 6GB | Full screenshots + view hierarchies |
| `ui_metadata` | 2MB | Metadata CSV |
| `ui_vectors` | 8MB | 64-dim layout vectors |

### Rico Semantic Categories (26 total)

- **Containers**: Card, Modal, Web View
- **Navigation**: Toolbar, Navigation Bar, Bottom Navigation, Tab Bar, Drawer, Multi-Tab, Pager Indicator
- **Content**: List Item, Text, Image, Icon, Advertisement
- **Controls**: Button, Text Button, Checkbox, Radio Button, Switch, Input, Slider, Spinner, Date Picker, Number Stepper

---

## Complexity Metrics

Metrics for ranking layout complexity in the corpus:

| Metric | Description | Weight |
|--------|-------------|--------|
| `element_count` | Total number of nodes | 0.25 |
| `max_depth` | Deepest nesting level | 0.20 |
| `orientation_mix` | H/V/Overlay diversity | 0.15 |
| `type_diversity` | Unique component types used | 0.15 |
| `density` | Elements per unit area | 0.15 |
| `symmetry` | Balance of element distribution | 0.10 |

> Complexity normalized to [0, 1] using min-max scaling across the corpus.

---

## References

- [Rico Dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/)
- [Enrico Dataset](https://github.com/luileito/enrico)
- [WebSight Dataset](https://huggingface.co/datasets/HuggingFaceM4/WebSight)
- [EGFE Dataset](https://zenodo.org/records/8004165)
- [ShowUI Desktop](https://huggingface.co/datasets/Voxel51/ShowUI_desktop)
