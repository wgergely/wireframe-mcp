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

---

## Vector Database Matrix for UI Layouts

Investigated vector databases suitable for storing UI layout embeddings (from JSON trees) and enabling hybrid search for LLM-MCP agents.

### Comparison Matrix

| Feature | **Qdrant** | **Chroma** | **Weaviate** | **Milvus** | **pgvector** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Type** | Native Vector Search Engine | AI-native Embedding DB | Vector Search Engine | Vector Database | PostgreSQL Extension |
| **Language** | Rust | Python/Rust | Go | Go | C |
| **JSON Support** | **Excellent**. JSON payloads. Deep filtering `must`, `should` with nested logic. | **Good**. Metadata dictionary. Mongo-style filtering `$eq`, `$and`. | **Good**. Stores objects with schema. | **Excellent**. JSON data type support. | **Good**. JSONB in Postgres. |
| **Hybrid Search** | **Strong**. Native dense + sparse vectors. RRF fusion. | **Native**. Dense + Keyword. Simple API. | **Leading**. Built-in BM25 + Vector + Metadata. Configurable alpha. | **Supported**. via multi-vector search. | **Manual**. Combine SQL TSVECTOR + Vector. |
| **Deployment** | Docker, Cloud, **Local Mode** (Disk/Mem) | **Local (In-process)**, Client/Server | Docker, Cloud, Embedded (experimental) | Docker, K8s, Cloud (Heavyweight) | Standard Postgres |
| **Ecosystem** | Strong Python SDK, LangChain, LlamaIndex | Very strong Python integration. Simple API. | Strong ecosystem. | Enterprise focused. | Standard SQL tooling. |
| **Suitability** | **High**. Great balance of performance and structure. | **Medium/High**. Best for prototyping/local agents. | **High**. robust for complex schemas. | **Medium**. Overkill for local corpus generally. | **High**. If already using Postgres. |

### Suitability Analysis for UI Layouts

Storing UI layouts (Rico/WebUI) requires handling:
1.  **Hierarchical Data**: Trees of components.
2.  **Metadata**: Bounding boxes, types, text content.
3.  **Embeddings**: Semantic vector of the screen or component.

**Key Findings:**
*   **Qdrant**: Best balance. "Payloads" are perfect for storing the raw JSON UI tree (or a simplified version) alongside the vector. The filtering is powerful enough to query "Screens with 'Login Button' AND 'Red Background'".
*   **Chroma**: Excellent for "Embeddings first" workflows. If the goal is just "Find similar screens", it's the easiest to start with.
*   **Weaviate**: Strong if defining a strict schema for UI components (e.g. classes for `Button`, `Container`) is desired.

### Recommendations for LLM-MCP Usage

1.  **Primary Recommendation: Qdrant**
    *   **Why**: It can run locally (Docker) or in-memory for testing, has a highly performant Rust core, and its "payload" system is ideal for unstructured JSON UI trees. The hybrid search is sophisticated (RRF) which helps when mixing "text description" search with "visual layout" search.
    *   **MCP Fit**: Python client is excellent. Easy to package in a Docker composition with the MCP server.

2.  **Secondary Recommendation: Chroma**
    *   **Why**: Zero-setup (runs in-process). If the corpus is manageable (<1M screens) and runs directly in the Agent's environment, this is the lowest friction path.

3.  **Alternative: pgvector**
    *   **Why**: If the project already needs a relational DB for other corpus metadata, adding pgvector is logical to keep a single store.
