# Layout Broker: Architecture & Implementation Specification

## 1. Abstract & Research Basis

This document defines the architecture for the **Layout Broker**, a middleware system designed to facilitate high-fidelity UI generation from Natural Language (NL).

The system addresses the **"Stochastic Gap"** in Large Language Model (LLM) outputs—the phenomenon where direct code generation (e.g., producing raw HTML/React) leads to structural hallucinations and non-deterministic layouts. Instead of direct generation, this architecture implements a **Compiler Pipeline Pattern**:

1. **Lexing/Parsing (LLM):** Translates NL into a strict Intermediate Representation (IR).

2. **Validation (Python):** Enforces logical constraints (containment, accessibility).

3. **Code Generation (D2/Kroki):** Deterministically renders the validated IR for user review and feedback.

### Supporting Research

* **Structured Representations vs. Natural Language:** Research indicates that constraining LLM output to a structured schema (JSON/IR) significantly outperforms free-form generation. *Chen et al. (2025)* demonstrated that structured interfaces improve user preference win rates by **~30%** over conversational baselines by enabling "generation-evaluation cycles" [1].

* **Constraint-Based DSLs:** The use of Domain Specific Languages (DSLs) to restrict the "program space" of the LLM is a proven technique to reduce hallucination. *UIFormer (Dec 2025)* validates that restricting LLM output to a DSL reduces token consumption by **~50%** and improves semantic accuracy by ensuring all outputs are "physically viable" [2][3].

## 2. System Architecture

The system follows a standard **Three-Stage Compiler** design pattern.

```mermaid
graph LR
    subgraph Client
    NL[Natural Language Input]
    end

    subgraph "Stage 1: Frontend (LLM)"
    Lexer[LLM Tokenizer]
    Parser[Structured Output Parser]
    AST[Abstract Syntax Tree (JSON)]
    end

    subgraph "Stage 2: Middleware (MCP Server)"
    Validator[Static Analysis / Validation]
    Optimizer[Layout Normalizer]
    end

    subgraph "Stage 3: Backend (Renderer)"
    CodeGen[Transpiler -> D2 DSL]
    Render[Kroki Rendering Engine]
    end

    NL --> Lexer
    Lexer --> Parser
    Parser --> AST
    AST --> Validator
    Validator --> Optimizer
    Optimizer --> CodeGen
    CodeGen --> Render
```

### Stage 1: The Semantic Parser (LLM)

* **Role:** Acts as the parser, converting unstructured intent into a structured Abstract Syntax Tree (AST).

* **Constraint Mechanism:** Utilizes "Context-Free Grammars" (via JSON Schema) to enforce valid syntax during the decoding phase. This prevents the generation of syntactically invalid trees.

### Stage 2: The Middleware (Python MCP)

* **Role:** Performs static analysis on the AST.

* **Operations:**

  * **Cycle Detection:** Ensuring no node is an ancestor of itself.

  * **Orphan Removal:** Pruning nodes that are not connected to the root `LayoutRequest`.

  * **Type Checking:** Validating that properties (e.g., `flex_ratio`) fall within allowed numeric ranges.

### Stage 3: The Renderer (D2/Kroki)

* **Role:** Deterministic transpilation.

* **Choice of D2:** Selected over PlantUML for its **Constraint-Based Layout Engine**. D2 explicitly models "containers" and "connections" separately, which aligns 1:1 with modern CSS Flexbox/Grid systems, whereas PlantUML relies on older heuristic placement algorithms.

### 2.1 Execution Strategies (Passive vs. Agentic)

Your architecture supports two distinct execution flows. The **Agentic** flow is recommended for centralized control.

* **Passive Mode (Standard MCP):** The Client controls the LLM. The Client calls the LLM, the LLM decides to call your Tool, your Tool validates.

  * *Pros:* Users can bring their own API keys.

  * *Cons:* Harder to enforce strict prompting strategies.

* **Agentic Mode (Orchestrator):** The Client sends NL to your MCP Server. Your Server *internally* calls the LLM API (Stage 1) and returns the final result.

  * *Pros:* You fully control the system prompt, model selection, and retry logic (e.g., auto-fixing invalid JSON).

  * *Cons:* Higher server costs (you pay for API tokens).

### 2.2 Model Selection Strategy

The "Stage 1" Lexer requires specific model capabilities.

* **Frontier General Models (Recommended):**

  * **Claude 3.5 Sonnet:** Currently SOTA for coding and complex instruction following. Highly recommended for the initial "Layout Synthesis" step due to superior spatial reasoning.

  * **OpenAI o1 / GPT-4o:** Strong reasoning capabilities, excellent for validating complex user logic before layout generation.

* **Specialized Coding Models:**

  * **DeepSeek-Coder-V2 / DeepSeek-V3:** Specialized open-weights models trained heavily on code and configuration files. These are extremely cost-effective alternatives to Claude/GPT-4o for the strict "JSON-to-JSON" translation tasks.

  * **Qwen 2.5 Coder:** Another high-performance option for strict schema adherence.

## 3. Data Transformation Lifecycle (The "Paper Trail")

This section explicitly maps how data mutates from "Fuzzy Intent" to "Rigid Syntax." This answers the question: *How does the translation actually happen?*

### Step 1: User Intent (Input)

```text
User: "I need a dashboard with a sidebar on the left and a main content area."
```

### Step 2: LLM Output (The "Unboxing")

The LLM does **NOT** generate D2 code. It fills a JSON template. The "Black Box" is forced to conform to this shape via Tool Calling.

```json
{
  "root": {
    "id": "root_container",
    "type": "container",
    "orientation": "horizontal",
    "children": [
      {
        "id": "sidebar",
        "type": "sidebar",
        "flex_ratio": 3,
        "label": "Navigation"
      },
      {
        "id": "main_content",
        "type": "container",
        "flex_ratio": 9,
        "label": "Dashboard Widgets"
      }
    ]
  }
}
```

### Step 3: Middleware Transpilation (The "Constraint")

Your Python code (Stage 3) consumes the JSON and algorithmically writes the D2. **This guarantees valid syntax.** The LLM never sees D2.

```d2
root_container: container {
  direction: right
  
  sidebar: Navigation {
    width: 30%
  }
  
  main_content: Dashboard Widgets {
    width: 70%
  }
}
```

## 4. Data Structure Specification (The IR)

The Intermediate Representation is the contract between the LLM and the Runtime. It is defined formally using **Pydantic**, which allows for runtime type reflection and automatic JSON Schema generation.

### 4.1 Type Definitions

```python
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, conint

class Orientation(str, Enum):
    """
    Defines the flow direction of a container. 
    Maps to CSS flex-direction.
    """
    HORIZONTAL = "horizontal"  # flex-row
    VERTICAL = "vertical"      # flex-col
    OVERLAY = "overlay"        # z-index stacking

class ComponentType(str, Enum):
    """
    Restricted vocabulary of UI primitives.
    This limits the LLM's 'creative' space to known-renderable entities.
    """
    CONTAINER = "container"
    BUTTON = "button"
    INPUT = "input"
    DATAGRID = "datagrid"
    NAVBAR = "navbar"
    SIDEBAR = "sidebar"
    TEXT = "text"
    IMAGE = "image"

class LayoutNode(BaseModel):
    """
    Recursive node definition for the UI AST.
    """
    id: str = Field(..., description="Unique immutable identifier (UUID v4 recommended)")
    type: ComponentType
    label: Optional[str] = Field(None, description="Human-readable text content")
    
    # Layout Constraints
    flex_ratio: conint(ge=1, le=12) = Field(
        default=1, 
        description="Grid span ratio (1-12 standard grid system)"
    )
    
    # Recursive Definition
    children: List['LayoutNode'] = Field(
        default_factory=list,
        description="Nested child nodes. Strictly strictly hierarchical."
    )
    orientation: Orientation = Field(
        default=Orientation.VERTICAL, 
        description="Layout flow for immediate children"
    )

    class Config:
        # Ensures recursive types are handled correctly in JSON Schema export
        use_enum_values = True
```

## 5. Implementation Protocol

### 5.1 MCP Tool Interface

The server exposes a single primitive tool: `generate_layout`.

**Tool Definition:**

```json
{
  "name": "generate_layout",
  "description": "Compiles a natural language description into a UI Layout AST.",
  "input_schema": {
    "type": "object",
    "properties": {
      "root": { "$ref": "#/definitions/LayoutNode" },
      "meta_description": { "type": "string" }
    },
    "required": ["root"]
  }
}
```

### 5.2 Transpilation Logic (Python -> D2)

The transpiler performs a depth-first traversal of the validated AST to emit D2 code.

* **Mapping:** `LayoutNode` -> D2 Class

* **Mapping:** `Orientation` -> D2 `direction` attribute

```python
def to_d2(node: LayoutNode) -> str:
    """
    Deterministic transpiler from IR to D2 DSL.
    """
    # D2 syntax for defining a shape
    shape_def = f'{node.id}: {node.label or node.type.value}'
    
    # Block start
    lines = [f"{shape_def} {{"]
    
    # Orientation Logic
    if node.orientation == Orientation.HORIZONTAL:
        lines.append("  direction: right")
    
    # Recursive Descent
    for child in node.children:
        child_str = to_d2(child)
        # Indent child content
        lines.append("  " + child_str.replace("\n", "\n  "))
        
    # Block end
    lines.append("}")
    return "\n".join(lines)
```
