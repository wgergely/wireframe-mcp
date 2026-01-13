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

---

## Dataset Suitability Ranking (2024 Analysis)

Ranked based on **Acquirability** (ease of download), **License** (freedom to use/train), and **Normalizability** (ease of converting to vector-ready JSON trees).

| Rank | Dataset | Availability & License | Normalizability | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Enrico** | **Excellent**. Direct ZIP download. **MIT License**. Hosted on GitHub/HuggingFace. | **High (5/5)**. Cleaned subset of Rico. JSON includes semantic annotations and design topics. Perfect for training/prototyping. | **🏆 Best Starter** |
| **2** | **Rico (Semantic)** | **High**. Available via HuggingFace (Voxel51/RootsAutomation). **CC-BY-4.0**. | **High (5/5)**. Standard Android View Hierarchy JSON. Rich semantic labels (Button, Text, Image). | **✅ Standard Corpus** |
| **3** | **WebUI** | **Medium**. GitHub scripts to download 400k pages. Copyright ambiguous (likely fair use/research). | **Medium (3/5)**. HTML DOM is noisier than Android JSON. Requires significant cleaning to extract "visual layout" tree. | **Example of Quantity** |
| **4** | **ScreenSpot** | **High**. HuggingFace. **Apache 2.0**. | **Low (2/5)**. Focused on "Instructions + Bounding Box". Lacks deep hierarchical layout trees for generation. Good for *grounding* (finding elements) but not *generating* them. | **Evaluation Only** |
| **5** | **Original Rico** | **Low**. Official site often restricted/deprecated. | **High (5/5)**. Same as Semantic but harder to get legally/technically. | **Use Derivatives** |

### Data Acquisition Strategy

1.  **Phase 1 (MVP)**: Download **Enrico** (1.4k screens).
    *   *Why*: Small, MIT licensed, pre-cleaned. Immediate "World Hello" for the Vector DB.
2.  **Phase 2 (Scale)**: Ingest **Rico Semantic** (~72k screens) from HuggingFace.
    *   *Why*: The standard for training. CC-BY-4.0 is safe for most internal agent usage.
3.  **Phase 3 (Web)**: Scrape/Download **WebUI** only if Android layouts prove insufficient for the User's needs.

---

3.  **Phase 3 (Web)**: Scrape/Download **WebUI** only if Android layouts prove insufficient for the User's needs.

---

## Top 10 Recommended Datasets for Wireframe Broker (React & Desktop Focus)

Ranked by suitability for generating **React Frontends** (Web) and **Python Desktop Interfaces** (PySide/Qt).

| Rank | Dataset | Focus | Scale | Why it Fits |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **WebSight** (HuggingFaceM4) | **React/Web** | 2M (Synthetic) | **HTML + Tailwind**. The closest "Figma-to-Code" dataset. Perfect for training models to generate clean, modern React code from visuals. |
| **2** | **ShowUI-Desktop** | **Desktop** | ~8k | **PC GUI Focus**. Specifically targets desktop applications. Critical for the "PySight/PySide" requirement where mobile datasets (Rico) fail. |
| **3** | **Rico (Semantic)** | **Layout Logic** | 72k | **Structural Logic**. Even though it's Android, the *graph* of "Container -> Button" is universal. Best for learning general UI composition. |
| **4** | **Enrico** | **Design** | 1.4k | **Curated Quality**. Good for understanding "Login Screens" vs "Profile Screens" visually. High quality standard. |
| **5** | **WebUI** (Anthropics) | **Web** | 400k | **Volume**. Massive raw data for web layouts, though requires filtering for "React-like" quality. |
| **6** | **ScreenSpot** | **Grounding** | 1.2k+ | **Multi-Platform**. Covers macOS/Windows. Excellent for evaluating if the generated layout *makes sense* to a user. |
| **7** | **Vision2UI** | **Real Web** | 2k+ | **Real-World Code**. Maps real screenshots to HTML code, bridging the gap between "Design" and "Implementation". |
| **8** | **MUI / AntD Docs** *(Custom)* | **Components** | N/A (Scrape) | **The "React Gold Standard"**. Not a zip file, but scraping the *examples* from Material UI/Ant Design docs provides the highest quality labelled {Image -> Code} pairs. |
| **9** | **Aria-UI** | **Context** | 7.8k | **Instruction Following**. Good for Agents that need to "Click the Submit button". |
| **10** | **Simulated PySide** *(Custom)* | **Desktop** | Synthetic | **Required for Desktop Code**. Since large PySide datasets don't exist, using a script to generate random standard Qt Layouts is the recommended path for code-training. |

### Recommendation for "Wireframe Broker"
*   **React Layer**: Use **WebSight** for visual-to-code mapping. Supplement with a custom scrape of **MUI/Ant Design** docs if high-fidelity code is needed.
*   **Desktop Layer**: Use **ShowUI-Desktop** for understanding PC layouts.
*   **Composition Layer**: Keep **Rico** as the backend vector store for abstract layout retrieval (e.g. "Find me a 3-column layout").

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
