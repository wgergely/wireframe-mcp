# Corpus Generation Research: UI Layout Datasets

Comprehensive catalog of available UI datasets for building a multi-platform corpus.

---

## Dataset Catalog

### Mobile UI Datasets

| Dataset | Size | Platform | Format | Key Features | Source |
|---------|------|----------|--------|--------------|--------|
| **Rico** | 72k screens | Android | JSON + PNG | 24 component categories, semantic annotations | [interactionmining.org](http://interactionmining.org/rico) |
| **Rico (Full)** | 66k UIs | Android | JSON + PNG | Complete view hierarchies, 6GB | [interactionmining.org](http://interactionmining.org/rico) |
| **Enrico** | 1.4k screens | Android | JSON | Rico subset with design topics | [HuggingFace](https://huggingface.co/datasets/luileito/enrico) |
| **ReDraw** | 191k components | Android | JSON + images | 15 GUI component categories | Research paper |

---

### Web UI Datasets

| Dataset | Size | Format | Key Features | Source |
|---------|------|--------|--------------|--------|
| **WebUI** | 400k pages | JSON + screenshots | Full-page screenshots, accessibility trees, bounding boxes, computed CSS | [GitHub](https://anthropics.github.io/webui/) |
| **WaveUI-25K** | 25k images | Annotated images | UI element annotations for automation | [Research](https://github.com) |
| **Vision2UI** | 20k samples | HTML + images | HTML code, UI designs, element positions (Common Crawl) | [arXiv](https://arxiv.org) |
| **WebCode2M** | 2M+ pages | HTML + images | Webpage code, DOM tree structure, bounding boxes | [arXiv](https://arxiv.org) |
| **UI-Elements-Detection** | 300+ sites | YOLO format | 1920x1080 screenshots, 15 UI classes | [Kaggle](https://kaggle.com) / [HuggingFace](https://huggingface.co) |
| **Roboflow Website Screenshots** | 1000+ sites | Auto-annotated | Buttons, headings, links, images, iframes | [Roboflow](https://roboflow.com) |
| **GroundUI-18K** | 18k images | Annotated | Instructions and bounding box annotations | Research |

---

### Desktop UI Datasets

| Dataset | Size | Platform | Format | Key Features | Source |
|---------|------|----------|--------|--------------|--------|
| **ScreenSpot** | Multi-platform | macOS/Windows/Mobile | Annotated | Element types, Apache 2.0 license | [HuggingFace](https://huggingface.co/datasets/ScreenSpot) |
| **ShowUI_desktop** | 7.5k screenshots | 15 apps | Annotated | 8k element annotations | GitHub |
| **Aria-UI Desktop** | 7.8k images | Ubuntu | Instructions | 150k instructions | [GitHub](https://github.com/Aria-UI) |
| **DeskVision** | Large-scale | Desktop | Auto-captioned | Rich annotations | Research |
| **Desktop-UI-Dataset** | 51 screenshots | Desktop | PASCAL VOC XML | CC BY license | [GitHub](https://github.com/waltteri/desktop-ui-dataset) |
| **Zenodo Desktop UI** | 100+ screenshots | Various | Annotated | PDF readers, CRM, email clients | [Zenodo](https://zenodo.org) |

---

### Design-Focused Datasets

| Dataset | Size | Format | Key Features | Source |
|---------|------|--------|--------------|--------|
| **EGFE** | High-quality | Sketch/Figma JSON | Layered design prototypes, metadata | [Zenodo](https://zenodo.org) |
| **gridaco/ui-dataset** | 20k+ screenshots | Annotated | 100k+ labeled buttons, NLP tokens | [GitHub](https://github.com/gridaco/ui-dataset) |
| **UISketch** | Hand-drawn | Sketches | UI sketches for recognition | Kaggle |

---

## Rico Dataset Details (Implemented)

**Download API:** `python -m src.corpus download-rico --dataset semantic`

### Available Subsets

| Subset | Size | Description |
|--------|------|-------------|
| `semantic` ✅ | 150MB | Semantic annotations (72k screens) |
| `ui_screenshots` | 6GB | Full screenshots + view hierarchies |
| `ui_metadata` | 2MB | Metadata CSV |
| `ui_vectors` | 8MB | 64-dim layout vectors |

### Rico Semantic Categories (24 total)

From `component_legend.json`:
- **Containers**: Card, Modal, Web View
- **Navigation**: Toolbar, Navigation Bar, Bottom Navigation, Tab Bar, Drawer, Multi-Tab, Pager Indicator
- **Content**: List Item, Text, Image, Icon, Advertisement
- **Controls**: Text Button, Checkbox, Radio Button, Switch, Input, Slider, Spinner, Date Picker, Number Stepper

---

## Recommended Datasets for Wireframe Broker

| Priority | Dataset | Rationale |
|----------|---------|-----------|
| **1** | Rico (semantic) ✅ | Already integrated, rich Android layouts |
| **2** | WebUI | 400k web pages with CSS/bounding boxes |
| **3** | ScreenSpot | Cross-platform (macOS/Windows/mobile) |
| **4** | Vision2UI | HTML DOM structure for web layouts |

---

## Complexity Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| `element_count` | Total number of nodes | 0.25 |
| `max_depth` | Deepest nesting level | 0.20 |
| `orientation_mix` | H/V/Overlay diversity | 0.15 |
| `type_diversity` | Unique component types used | 0.15 |
| `density` | Elements per unit area | 0.15 |
| `symmetry` | Balance of element distribution | 0.10 |

> [!IMPORTANT]
> Complexity normalized to [0, 1] using min-max scaling across the corpus.

---

## References

- Rico Dataset: http://interactionmining.org/rico
- WebUI Dataset: https://anthropics.github.io/webui/
- ScreenSpot: https://huggingface.co/datasets/ScreenSpot
- UI Complexity Metrics: ResearchGate studies on GUIEvaluator/GUIExaminer
