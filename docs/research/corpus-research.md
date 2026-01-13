# Corpus Generation Research: UI Layout Database

Research findings and architectural design for building a corpus generation system that ingests external UI datasets (Rico, ReDraw), maps them to our IR format, computes normalized complexity scores, and provides progressive retrieval from simple to complex layouts.

---

## Source Datasets

| Dataset | Size | Format | Key Features |
|---------|------|--------|--------------|
| **Rico** | 72k screens | JSON hierarchies | 24 component categories, 197 button concepts, 97 icon classes |
| **ReDraw** | 191k components | JSON + images | 15 GUI component categories, Android-focused |
| **WebUI** | 400k pages | JSON + screenshots | Web layouts, metadata extraction |

### Rico Dataset Details

- **Download**: http://interactionmining.org/rico
- **Semantic Annotations**: 150MB subset with component labels
- **JSON Structure**: `activity.root.children[*]` with `class`, `bounds`, `componentLabel`

### Rico Semantic Component Categories (24 total)

From `component_legend.json`:
- List Item, Text, Image, Icon, Text Button, Card
- Toolbar, Navigation Bar, Bottom Navigation, Tab Bar
- Drawer, Modal, Checkbox, Radio Button, Switch
- Input, Slider, Spinner, Pager Indicator, Web View
- Advertisement, Multi-Tab, Date Picker, Number Stepper

---

## Complexity Metrics (from HCI Research)

| Metric | Description | Weight |
|--------|-------------|--------|
| `element_count` | Total number of nodes | 0.25 |
| `max_depth` | Deepest nesting level | 0.20 |
| `orientation_mix` | H/V/Overlay diversity | 0.15 |
| `type_diversity` | Unique component types used | 0.15 |
| `density` | Elements per unit area | 0.15 |
| `symmetry` | Balance of element distribution | 0.10 |

> [!IMPORTANT]
> Complexity normalized to [0, 1] using min-max scaling across the corpus. A single empty container = 0.0, a masonry grid with flyouts = ~1.0.

### Complexity Examples

| Layout Description | Expected Complexity |
|--------------------|---------------------|
| Single empty container | 0.0 |
| Container with 2 buttons | 0.05 - 0.10 |
| Simple form (5 inputs vertical) | 0.15 - 0.25 |
| Sidebar + content area | 0.25 - 0.35 |
| Dashboard with header + sidebar + grid | 0.40 - 0.55 |
| Multi-tab navigation + nested lists | 0.60 - 0.75 |
| Masonry grid + flyout panels | 0.80 - 0.95 |
| Complex editor with toolbars + panels + tabs | 0.95 - 1.0 |

---

## Proposed Module Structure

```
src/corpus/
├── __init__.py           # Public API exports
├── lib.py                # CorpusManager, retrieval APIs
├── complexity.py         # ComplexityCalculator, normalization
├── ingest/
│   ├── __init__.py
│   ├── rico.py           # Rico dataset ingestion
│   └── base.py           # DatasetIngester ABC
├── models.py             # CorpusEntry, Metadata models
├── db.py                 # SQLite storage for corpus
└── test.py               # Unit tests
```

---

## Core API Design

### CorpusManager

```python
class CorpusManager:
    def download_rico(self, target: Path) -> None: ...
    def ingest_dataset(self, ingester: DatasetIngester, source: Path) -> int: ...
    def get_by_complexity(self, min_c: float, max_c: float) -> list[CorpusEntry]: ...
    def get_progressive_set(self, steps: int = 10) -> list[list[CorpusEntry]]: ...
    def validate_corpus(self) -> list[ValidationError]: ...
    def stats(self) -> dict: ...
```

### Database Schema

```sql
CREATE TABLE entries (
    id TEXT PRIMARY KEY,
    layout_json TEXT NOT NULL,
    complexity REAL NOT NULL,
    source TEXT NOT NULL,
    category TEXT,
    metadata_json TEXT
);

CREATE INDEX idx_complexity ON entries(complexity);
```

---

## References

- Rico Dataset: http://interactionmining.org/rico
- UI Complexity Metrics: ResearchGate studies on GUIEvaluator/GUIExaminer
- Hugging Face Rico: https://huggingface.co/datasets/luileito/enrico
