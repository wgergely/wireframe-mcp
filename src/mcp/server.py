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
    get_server_capabilities,
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
# Server Instance
# =============================================================================

mcp = FastMCP(
    name="wireframe-mcp",
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


# =============================================================================
# Discovery Tools
# =============================================================================


@mcp.tool
def ping() -> dict:
    """Check if the server is running.

    Returns:
        Server status with version.
    """
    return {
        "status": "ok",
        "version": get_server_version(),
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
    logger.info(f"Starting wireframe-mcp server v{get_server_version()}")
    logger.info(f"Transport: {transport.value}")

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
