"""FastMCP server instance for wireframe-mcp.

This module provides the MCP server that exposes layout generation tools
to LLM clients. The API is intentionally minimal - focused on the core
user workflow:

    1. generate_layout: NL description → layout JSON + draft text tree
    2. preview_layout: Layout → visual wireframe (PNG/SVG)
    3. generate_variations: Create N layout options for comparison

Usage:
    # STDIO mode (for Claude Desktop)
    python -m src.mcp.server

    # HTTP mode (for web deployment)
    python -m src.mcp.server --transport http --port 18080

    # Via CLI
    python . mcp run
    python . mcp serve --port 18080
"""

import argparse
import json
import logging
import sys
from functools import lru_cache
from typing import Any

from fastmcp import FastMCP

from .lib import (
    TransportType,
    get_server_version,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Helpers (Internal)
# =============================================================================


def _get_valid_providers() -> frozenset[str]:
    """Get valid provider names from registry."""
    from src.providers import list_providers

    return frozenset(list_providers())


def _validate_provider(provider: str) -> None:
    """Validate provider parameter against registry."""
    valid = _get_valid_providers()
    if provider not in valid:
        raise ValueError(f"Invalid provider '{provider}'. Valid: {sorted(valid)}")


def _validate_temperature(temperature: float) -> None:
    """Validate temperature parameter."""
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"Temperature must be 0.0-2.0, got {temperature}")


# =============================================================================
# Server Instructions (LLM Guidance)
# =============================================================================

SERVER_INSTRUCTIONS = """\
## Wireframe MCP Server

Generates UI wireframe layouts from natural language. Prevents implementation
miscommunication by validating understanding BEFORE coding begins.

### Quick Start
1. `status()` → check readiness
2. `generate_layout("your UI description")` → get draft
3. Review draft with user
4. `preview_layout(layout)` → visual wireframe (optional)
5. `validate_layout(layout)` → check before implementation

### Complete Workflow

**Creating Layouts:**
- `generate_layout(query)` - Single layout from description
- `generate_variations(query, count=3)` - Multiple options to compare
- `refine_layout(artifact_id, feedback)` - Iterate on existing layout

**Reviewing:**
- `get_artifact(id)` - Get full layout from artifact ID
- `search_layouts(query)` - Find similar examples in corpus
- `validate_layout(layout)` - Check for issues

**Preview:**
- `preview_layout(layout)` - Render visual wireframe

### Iteration Example
```
User: "Create a dashboard"
→ generate_layout("dashboard with sidebar")
→ User: "Make the sidebar narrower"
→ refine_layout(artifact_id, "make sidebar narrower")
→ User: "Perfect, show me the wireframe"
→ preview_layout(layout)
→ User: "Approved!"
→ validate_layout(layout)  # Check before implementation
```

### Service Status
| Service | Required For | If Missing |
|---------|-------------|------------|
| LLM provider | Generation | Set API key in .env |
| Kroki | Preview | Run: python . docker up |
| RAG index | Context | Run: python . dev index build |

### Getting Help
- `help()` - List topics
- `help('workflow')` - Step-by-step guide
- `help('troubleshooting')` - Common issues
"""

# =============================================================================
# Server Instance
# =============================================================================

mcp = FastMCP(
    name="wireframe-mcp",
    instructions=SERVER_INSTRUCTIONS,
)


# =============================================================================
# Core LLM Tools (User Workflow)
# =============================================================================


@mcp.tool
def generate_layout(
    query: str,
    model: str | None = None,
    temperature: float = 0.7,
    session_id: str | None = None,
    parent_id: str | None = None,
) -> dict[str, Any]:
    """Generate a UI layout from natural language description.

    This is the primary tool. Returns structured layout JSON and a
    human-readable draft for quick review.

    Args:
        query: Natural language description of the desired layout.
            Examples:
            - "login form with email and password"
            - "dashboard with sidebar and stats cards"
            - "settings page with toggle switches"
        model: LLM model to use (optional, uses default if not specified).
        temperature: Creativity level 0.0-2.0. Default: 0.7
        session_id: Group related generations together (optional).
        parent_id: Link to previous layout for refinements (optional).

    Returns:
        Dictionary with:
        - artifact_id: Unique ID to reference this layout later
        - layout: The generated layout structure as JSON
        - draft: Human-readable text tree for quick review:
            ```
            Dashboard [container]
            ├── Sidebar [nav]
            │   ├── Logo [image]
            │   └── Menu [list]
            └── Main [panel]
                └── Stats [card]
            ```

    Example workflow:
        1. generate_layout("login form") → review draft
        2. User: "add forgot password link"
        3. generate_layout("login form with forgot password", parent_id=...)
    """
    _validate_temperature(temperature)

    from .tools.generate import generate_layout as _generate

    return _generate(
        query=query,
        model=model,
        temperature=temperature,
        include_rag=True,
        persist=True,
        session_id=session_id,
        parent_id=parent_id,
    )


@mcp.tool
def preview_layout(
    layout: dict[str, Any],
    output_format: str = "png",
) -> dict[str, Any]:
    """Render a layout to a visual wireframe image.

    Use this when the user wants to SEE the layout, not just
    read the draft text tree.

    Args:
        layout: Layout JSON from generate_layout.
        output_format: "png" or "svg". Default: "png"

    Returns:
        Dictionary with:
        - image_data: Base64-encoded image
        - format: Image format used
        - size_bytes: Image size

    Note: Requires rendering service (Kroki) to be running.
    """
    from .tools.preview import preview_layout as _preview

    return _preview(layout=layout, style="wireframe", output_format=output_format)


@mcp.tool
def generate_variations(
    query: str,
    count: int = 3,
    model: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Generate multiple layout variations for comparison.

    Use this when the user wants OPTIONS to choose from.
    Creates N different layouts with varying creativity levels.

    Args:
        query: Natural language description of the desired layout.
        count: Number of variations (1-5). Default: 3
        model: LLM model to use (optional).
        session_id: Group with related generations (optional).

    Returns:
        Dictionary with:
        - variation_set_id: ID for this set of variations
        - variations: List of layout options, each with:
            - artifact_id: Unique ID for this variation
            - draft: Text tree preview (first 200 chars)
            - temperature: Creativity level used
        - diversity_score: How different the options are (0-1)

    Example:
        User: "Give me 3 different dashboard layouts"
        → Returns 3 variations ranked by complexity
        → Show drafts to user, preview the ones they like
    """
    if not 1 <= count <= 5:
        raise ValueError(f"count must be 1-5, got {count}")

    from .tools.generate import generate_variations as _generate_variations

    result = _generate_variations(
        query=query,
        count=count,
        model=model,
        include_rag=True,
        session_id=session_id,
    )

    # Simplify response for LLM consumption
    return {
        "variation_set_id": result["variation_set_id"],
        "variations": [
            {
                "artifact_id": a["id"],
                "draft": a["draft_preview"],
                "temperature": a["temperature"],
            }
            for a in result["artifacts"]
        ],
        "diversity_score": result["diversity_score"],
    }


@mcp.tool
def get_artifact(
    artifact_id: str,
    include_lineage: bool = False,
) -> dict[str, Any]:
    """Retrieve a previously generated layout by its artifact ID.

    Use this after generate_variations to get the full layout for
    a specific variation you want to preview.

    Args:
        artifact_id: The artifact UUID from generate_layout or variations.
        include_lineage: Include parent/child relationships. Default: False

    Returns:
        Dictionary with:
        - artifact: Full artifact data including layout, draft, stats
        - lineage: Parent/child IDs (if include_lineage=True)

    Example workflow:
        1. generate_variations("dashboard", count=3) → get artifact IDs
        2. User picks variation 2
        3. get_artifact(artifact_id) → get full layout
        4. preview_layout(layout) → render wireframe
    """
    from .tools.history import get_artifact as _get_artifact

    return _get_artifact(artifact_id=artifact_id, include_lineage=include_lineage)


@mcp.tool
def validate_layout(
    layout: dict[str, Any],
) -> dict[str, Any]:
    """Validate a layout structure before implementation.

    Use this before handing off to a coding agent to catch issues early.
    Checks structure, constraints, and accessibility.

    Args:
        layout: Layout JSON from generate_layout.

    Returns:
        Dictionary with:
        - valid: True if layout passes all checks
        - errors: List of critical issues that must be fixed
        - warnings: List of non-critical suggestions
        - stats: Layout statistics (node count, depth, etc.)

    Example:
        >>> result = validate_layout(layout)
        >>> if not result["valid"]:
        ...     print("Fix these:", result["errors"])
    """
    from .tools.validate import validate_layout as _validate

    return _validate(layout=layout)


@mcp.tool
def search_layouts(
    query: str,
    k: int = 5,
) -> dict[str, Any]:
    """Search for similar layouts in the corpus.

    Use this to find inspiration or see what layouts exist
    for similar UI patterns.

    Args:
        query: Natural language description to search for.
        k: Number of results (1-20). Default: 5

    Returns:
        Dictionary with:
        - results: List of similar layouts with scores
        - total_in_index: Total layouts in the index

    Note: Requires RAG index to be built.
    """
    if not 1 <= k <= 20:
        raise ValueError(f"k must be 1-20, got {k}")

    from .tools.search import search_layouts as _search

    return _search(query=query, k=k)


@mcp.tool
def refine_layout(
    artifact_id: str,
    feedback: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Refine an existing layout based on feedback.

    Use this when the user wants to MODIFY an existing layout
    rather than generate a completely new one.

    Args:
        artifact_id: ID of the layout to refine.
        feedback: What to change (e.g., "move sidebar to right").
        model: LLM model to use (optional).
        temperature: Creativity level 0.0-2.0. Default: 0.7

    Returns:
        Dictionary with new layout and parent_id link.

    Example:
        User: "I like this but move the sidebar to the right"
        -> refine_layout(artifact_id, "move sidebar to right")
    """
    _validate_temperature(temperature)

    from .tools.refine import refine_layout as _refine

    return _refine(
        artifact_id=artifact_id,
        feedback=feedback,
        model=model,
        temperature=temperature,
    )


# =============================================================================
# Session History Tools
# =============================================================================


@mcp.tool
def get_session_history(
    session_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get session history for conversation reconstruction.

    Returns a chronological timeline of all tool calls and generated
    artifacts. Use this to understand what has happened in a design
    conversation or to resume work after context loss.

    Args:
        session_id: Session to get history for (uses current if None).
        limit: Maximum events to return (1-200). Default: 50

    Returns:
        Dictionary with session_id, timeline, artifact_count, interaction_count
    """
    limit = max(1, min(200, limit))

    from .tools.session_history import get_session_history as _get_session_history

    return _get_session_history(
        session_id=session_id,
        limit=limit,
        include_artifacts=True,
    )


@mcp.tool
def get_artifact_history(
    artifact_id: str,
) -> dict[str, Any]:
    """Get the full refinement history of an artifact.

    Traces all refinements that led to this artifact and returns
    the feedback given at each step. Essential for understanding
    how a design evolved.

    Args:
        artifact_id: Artifact to get history for.

    Returns:
        Dictionary with lineage, feedback_chain, and interactions
    """
    from .tools.session_history import get_artifact_history as _get_artifact_history

    return _get_artifact_history(artifact_id=artifact_id)


# =============================================================================
# Status Tools
# =============================================================================


@mcp.tool
def status() -> dict[str, Any]:
    """Check server health and dependency status.

    Use this FIRST to verify the server is ready before generating layouts.
    Reports availability of all required services.

    Returns:
        Dictionary with:
        - status: "healthy", "degraded", or "unhealthy"
        - version: Server version
        - capabilities: Which tools will work
            - generate_layout: True if LLM providers available
            - preview_layout: True if Kroki service running
            - rag_context: True if RAG index built
        - services: Detailed status of each dependency
        - action_required: What to fix if degraded/unhealthy

    Example response when healthy:
        {
            "status": "healthy",
            "capabilities": {
                "generate_layout": true,
                "preview_layout": true,
                "rag_context": true
            }
        }

    Example response when degraded:
        {
            "status": "degraded",
            "capabilities": {
                "generate_layout": true,
                "preview_layout": false,  # Kroki not running
                "rag_context": false      # Index not built
            },
            "action_required": [
                "Start Kroki: python . docker up",
                "Build RAG index: python . dev index build"
            ]
        }
    """
    from .health import get_server_health

    health = get_server_health()
    result = health.to_dict()

    # Add action_required for LLM consumption
    actions = []
    if not health.can_generate:
        actions.append("Configure LLM: Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")
    if not health.can_preview:
        actions.append("Start Kroki: python . docker up")
    if not health.can_use_rag:
        actions.append("Build RAG index: python . dev index build")

    if actions:
        result["action_required"] = actions

    # Add next steps guidance
    from .health import HealthStatus

    if health.status == HealthStatus.HEALTHY:
        next_steps = [
            "Ready! Call generate_layout(query) to create a wireframe.",
            "Example: generate_layout('login form with email and password')",
        ]
    elif health.can_generate:
        next_steps = [
            "Generation available. Preview may be limited.",
            "Call generate_layout(query) to create a wireframe.",
        ]
    else:
        next_steps = [
            "Server not ready. Review action_required items above.",
            "Most common: Set OPENAI_API_KEY in .env file.",
        ]

    result["next_steps"] = next_steps

    return result


@mcp.tool
def help(topic: str | None = None) -> dict[str, Any]:
    """Get detailed help on using this server.

    Use this for on-demand guidance about specific topics.
    Call without arguments to see available topics.

    Args:
        topic: Help topic (optional). One of:
            - "workflow": Step-by-step usage guide
            - "troubleshooting": Common issues and fixes
            - "models": Available LLM models
            - "examples": Example queries and outputs

    Returns:
        Dictionary with help content for the requested topic,
        or list of available topics if none specified.
    """
    topics = {
        "workflow": {
            "title": "Workflow Guide",
            "content": """
## Typical Workflow

1. **Check Status First**
   Call `status()` to verify the server is ready.
   If status is not "healthy", report issues to the user.

2. **Generate Layout**
   Call `generate_layout(query)` with a natural language description.
   Example: "login form with email, password, and remember me checkbox"

3. **Review Draft**
   The response includes a `draft` text tree for quick review:
   ```
   LoginForm [form]
   ├── Email [input]
   ├── Password [input]
   ├── Remember [checkbox]
   └── Submit [button]
   ```

4. **Preview (Optional)**
   Call `preview_layout(layout)` to render a visual wireframe.
   Returns base64-encoded PNG or SVG.

5. **Iterate**
   Use `parent_id` parameter to link refinements:
   `generate_layout("add forgot password link", parent_id="...")`

6. **Compare Options**
   Use `generate_variations(query, count=3)` to get multiple options.
""",
        },
        "troubleshooting": {
            "title": "Troubleshooting Guide",
            "content": """
## Common Issues

### "No LLM providers configured"
- **Cause**: No API keys set in environment
- **Fix**: User must set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file
- **Check**: Call `status()` to see which providers are available

### "Kroki not responding"
- **Cause**: Kroki Docker container not running
- **Fix**: Run `python . docker up` to start services
- **Note**: Generation still works, only preview is unavailable

### "RAG index not found"
- **Cause**: Vector index not built
- **Fix**: Run `python . dev index build`
- **Note**: Generation still works, but without context enhancement

### Layout generation fails
- **Check**: Is status "healthy" or "degraded"?
- **Check**: Is the query clear and specific?
- **Try**: Simpler query first, then add complexity

### Preview returns error
- **Check**: Is Kroki service running? (`status()` shows this)
- **Check**: Is the layout valid JSON from generate_layout?
""",
        },
        "models": {
            "title": "Available Models",
            "content": _get_models_help(),
        },
        "examples": {
            "title": "Example Queries",
            "content": """
## Example Queries

### Simple Components
- "login form with email and password"
- "navigation bar with logo and menu items"
- "card with image, title, and description"

### Complex Layouts
- "dashboard with sidebar navigation and main content area with stats cards"
- "settings page with sections for profile, notifications, and privacy"
- "e-commerce product page with image gallery, details, and add to cart"

### Specific Requirements
- "mobile-friendly contact form with validation hints"
- "two-column layout with left sidebar 200px wide"
- "header with centered logo and right-aligned user menu"

### Refinements (use with parent_id)
- "add a search box to the header"
- "make the sidebar collapsible"
- "add loading states to the buttons"
""",
        },
    }

    if topic is None:
        return {
            "available_topics": list(topics.keys()),
            "usage": "Call help(topic='workflow') for detailed guidance",
        }

    if topic not in topics:
        return {
            "error": f"Unknown topic: {topic}",
            "available_topics": list(topics.keys()),
        }

    return topics[topic]


def _get_models_help() -> str:
    """Generate models help content dynamically."""
    try:
        from src.config import get_available_llm_providers
        from src.providers import list_providers

        available = get_available_llm_providers()
        all_providers = list_providers()

        lines = ["## Available LLM Models", ""]

        if available:
            lines.append(f"**Configured providers**: {', '.join(available)}")
            lines.append("")
            lines.append("You can specify a model in `generate_layout(model='...')`.")
            lines.append("If not specified, the default model is used.")
        else:
            lines.append("**No providers configured**")
            lines.append("")
            lines.append("Set one of these API keys in .env:")
            lines.append("- OPENAI_API_KEY")
            lines.append("- ANTHROPIC_API_KEY")
            lines.append("- DEEPSEEK_API_KEY")

        lines.append("")
        lines.append(f"**Supported providers**: {', '.join(all_providers)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error loading model info: {e}"


@mcp.tool
def list_models() -> dict[str, Any]:
    """List available LLM models for layout generation.

    Returns which models are configured and can be used
    with generate_layout(model='...').

    Returns:
        Dictionary with:
        - available: List of configured provider names
        - default: The default model used if none specified
        - supported: All supported providers (may need API keys)
    """
    from src.config import get_available_llm_providers

    try:
        from src.providers import list_providers

        all_providers = list_providers()
    except Exception:
        all_providers = ["openai", "anthropic", "deepseek", "qwen"]

    available = get_available_llm_providers()

    return {
        "available": available,
        "default": available[0] if available else None,
        "supported": all_providers,
        "note": "Set API keys in .env to enable additional providers"
        if len(available) < len(all_providers)
        else "All supported providers are configured",
    }


# =============================================================================
# Resources (Schema Reference)
# =============================================================================


@lru_cache(maxsize=1)
def _cached_component_schema() -> str:
    """Cached component schema."""
    from src.schema import export_llm_schema

    return export_llm_schema()


@lru_cache(maxsize=1)
def _cached_layout_schema() -> str:
    """Cached layout schema."""
    from src.mid import LayoutNode

    schema = LayoutNode.model_json_schema()
    return json.dumps(schema, indent=2)


@mcp.resource("schema://components")
def get_component_schema() -> str:
    """Get the component type catalog.

    Returns JSON schema describing all UI component types
    (button, input, card, nav, etc.) with their properties.
    """
    return _cached_component_schema()


@mcp.resource("schema://layout")
def get_layout_schema() -> str:
    """Get the LayoutNode JSON schema.

    Returns the full schema for layout structures.
    """
    return _cached_layout_schema()


# =============================================================================
# Server Factory & Runner
# =============================================================================


def create_server() -> FastMCP:
    """Create and configure the MCP server instance.

    Returns:
        Configured FastMCP server instance.
    """
    return mcp


def run_server(
    transport: TransportType = TransportType.STDIO,
    host: str = "0.0.0.0",
    port: int = 18080,
) -> None:
    """Run the MCP server with specified transport.

    Args:
        transport: Transport type (stdio, http, sse).
        host: Bind address for HTTP/SSE.
        port: Port for HTTP/SSE.
    """
    from .health import log_startup_status

    logger.info(f"Starting wireframe-mcp server v{get_server_version()}")
    logger.info(f"Transport: {transport.value}")

    # Log health status on startup
    log_startup_status()

    if transport == TransportType.STDIO:
        logger.info("Running in STDIO mode (for Claude Desktop)")
        mcp.run()
    elif transport == TransportType.HTTP:
        logger.info(f"Running in HTTP mode at http://{host}:{port}/mcp")
        mcp.run(
            transport="http",
            host=host,
            port=port,
            path="/mcp",
        )
    elif transport == TransportType.SSE:
        logger.info(f"Running in SSE mode at http://{host}:{port}")
        mcp.run(
            transport="sse",
            host=host,
            port=port,
        )
    else:
        raise ValueError(f"Unknown transport: {transport}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for MCP server.

    Args:
        argv: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog="wireframe-mcp",
        description="MCP server for AI-assisted UI layout generation",
    )
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address for HTTP/SSE (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=18080,
        help="Port for HTTP/SSE (default: 18080)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    try:
        transport = TransportType(args.transport)
        run_server(
            transport=transport,
            host=args.host,
            port=args.port,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
