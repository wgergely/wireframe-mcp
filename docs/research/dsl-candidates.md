# Layout Broker Research Findings

## DSL Candidates Evaluated

### PlantUML Salt
- **Purpose**: GUI mockups/wireframes via text-based ASCII-like syntax
- **Strengths**:
  - High tolerance for minor syntax errors
  - Sketchy/low-fidelity look ideal for wireframes
  - Git-friendly text format
  - Supports buttons `[]`, radio `()`, checkboxes `[X]`, dropdowns `^`
  - Procedural features (`!if` for dynamic states)
- **Limitations**:
  - Limited layout/positioning control for complex GUIs
  - Styling constraints (skinparam partially supported)
  - Community described as "quiet", feature still maturing

### D2 (Declarative Diagramming)
- **Purpose**: Text-to-diagram with emphasis on software architecture
- **Strengths**:
  - First-class nested containers: `parent: { child }` or `parent.child`
  - Multiple layout engines: dagre, ELK, TALA (architecture-optimized)
  - Constraint support: `near`, `width`, `height`, per-container direction
  - Parent reference from child using `_` syntax
  - D2 Studio for GUI fine-tuning synced to text
- **Limitations**:
  - Medium tolerance (stricter syntax than PlantUML)
  - TALA engine is proprietary (dagre/ELK are open)

### Slint UI Toolkit
- **Purpose**: Production UI toolkit with markup language
- **Strengths**:
  - Live preview via VS Code extension, `slint-viewer --auto-reload`
  - Full spatial control and constraint system
  - SlintPad online editor for experimentation
- **Limitations**:
  - Designed for production UIs, not wireframes
  - Requires compilation/runtime for full features
  - Low tolerance for syntax errors (strict parser)
  - Overkill for structural "skeleton" purposes

### Excalidraw JSON
- **Purpose**: Hand-drawn style collaborative diagrams
- **Strengths**:
  - JSON format (LLM-friendly structured output)
  - `excalidraw-cli` for programmatic generation
  - `mermaid-to-excalidraw` converter available
  - High tolerance (schema is flexible)
- **Limitations**:
  - Manual spatial positioning (no auto-layout)
  - Requires headless browser for server-side PNG/SVG

---

## Rendering Backends

### Kroki (Recommended)
- Unified HTTP API for 25+ diagram types
- Supports D2, PlantUML, Excalidraw, Mermaid, GraphViz
- Self-hosted via Docker: `docker run -p 8000:8000 yuzutech/kroki`
- Output formats: SVG, PNG, PDF, JPEG
- MIT licensed, minimal resource requirements

### Direct CLI Tools
- D2: `d2 input.d2 output.svg`
- PlantUML: `java -jar plantuml.jar input.puml`
- Requires local installation of each tool

---

## MCP Server Implementation

### FastMCP Framework
- High-level Python framework for MCP servers
- Handles JSON-RPC protocol complexity
- Install: `pip install fastmcp`
- Decorator-based tool registration: `@mcp.tool()`

---

## Evaluation Matrix

| Criterion         | PlantUML Salt | D2     | Slint  | Excalidraw |
|-------------------|:-------------:|:------:|:------:|:----------:|
| Containment       | ✓             | ✓✓     | ✓✓     | ✓          |
| Hierarchy         | ✓             | ✓✓     | ✓✓     | ✓          |
| Spatial Intent    | Limited       | Strong | Full   | Manual     |
| LLM Tolerance     | High          | Medium | Low    | High       |
| Visual Clarity    | Sketchy       | Clean  | High   | Hand-drawn |
| Auto-Layout       | Basic         | Yes    | Yes    | No         |
| Kroki Support     | Yes           | Yes    | No     | Yes        |

**Recommendation**: D2 as default, PlantUML Salt as fallback provider.
