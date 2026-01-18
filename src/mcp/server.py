"""FastMCP server instance for wireframe-mcp.

This module provides the MCP server that exposes layout generation,
search, rendering, and validation tools to LLM clients.

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
import logging
import sys
from typing import Any

from fastmcp import FastMCP

from .lib import (
    TransportType,
    get_server_capabilities,
    get_server_version,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Server Instance
# =============================================================================

# Create the FastMCP server instance
mcp = FastMCP(
    name="wireframe-mcp",
)


# =============================================================================
# Health Check Tools
# =============================================================================


@mcp.tool
def ping() -> dict:
    """Check if the MCP server is running.

    Returns:
        Server status with version and capabilities.
    """
    return {
        "status": "ok",
        "version": get_server_version(),
        "capabilities": get_server_capabilities(),
    }


@mcp.tool
def get_server_info() -> dict:
    """Get detailed server information.

    Returns:
        Server name, version, and available features.
    """
    return {
        "name": "wireframe-mcp",
        "version": get_server_version(),
        "description": "MCP server for AI-assisted UI layout generation",
        "capabilities": get_server_capabilities(),
        "tools": [
            "ping",
            "get_server_info",
            "generate_layout",
            "validate_layout",
            "transpile_layout",
            "render_layout",
            "search_layouts",
        ],
        "resources": [
            "schema://components",
            "schema://layout",
            "config://models",
            "config://providers",
        ],
    }


# =============================================================================
# Layout Tools
# =============================================================================


@mcp.tool
def generate_layout(
    query: str,
    model: str | None = None,
    temperature: float = 0.7,
    provider: str = "d2",
    include_rag: bool = True,
) -> dict[str, Any]:
    """Generate a UI layout from natural language description.

    This is the primary creation tool. It returns structured JSON and a
    text tree representation for quick human review. Use render_layout
    to get a visual preview, or transpile_layout to get DSL code.

    Args:
        query: Natural language description of the desired layout.
            Examples:
            - "login form with email and password"
            - "dashboard with sidebar and main content area"
            - "settings page with toggle switches"
        model: LLM model to use. If not specified, uses default.
        temperature: Generation temperature (0.0-2.0). Default: 0.7
        provider: Target DSL provider ("d2", "plantuml"). Default: "d2"
        include_rag: Include similar layouts as context. Default: True

    Returns:
        Dictionary with:
        - layout: Generated LayoutNode as JSON
        - text_tree: Human-readable tree for quick review
        - stats: Generation statistics
    """
    from .tools.generate import generate_layout as _generate

    return _generate(
        query=query,
        model=model,
        temperature=temperature,
        provider=provider,
        include_rag=include_rag,
    )


@mcp.tool
def validate_layout(
    layout: dict[str, Any],
) -> dict[str, Any]:
    """Validate a layout structure for errors.

    Checks for structural issues like duplicate IDs, invalid flex ratios,
    cycles, and component-specific constraints.

    Args:
        layout: Layout JSON to validate.

    Returns:
        Dictionary with:
        - valid: Boolean indicating if layout passes all checks
        - errors: List of error objects
        - warnings: List of warning objects
        - stats: Structural statistics
    """
    from .tools.validate import validate_layout as _validate

    return _validate(layout=layout)


@mcp.tool
def transpile_layout(
    layout: dict[str, Any],
    provider: str = "d2",
) -> dict[str, Any]:
    """Convert a layout JSON to DSL code.

    Transpiles the layout to D2 or PlantUML diagram notation.

    Args:
        layout: Layout JSON to transpile.
        provider: Target DSL ("d2", "plantuml"). Default: "d2"

    Returns:
        Dictionary with:
        - dsl_code: The transpiled DSL code
        - provider: Provider used
        - line_count: Lines in output
    """
    from .tools.transpile import transpile_layout as _transpile

    return _transpile(layout=layout, provider=provider)


@mcp.tool
def render_layout(
    layout: dict[str, Any],
    format: str = "png",
    provider: str = "plantuml",
) -> dict[str, Any]:
    """Render a layout to an image.

    Converts the layout to DSL and renders via Kroki service.
    Requires Kroki to be running.

    Args:
        layout: Layout JSON to render.
        format: Output format ("png", "svg"). Default: "png"
        provider: DSL provider ("d2", "plantuml"). Default: "plantuml"

    Returns:
        Dictionary with:
        - image_data: Base64-encoded image
        - format: Image format
        - size_bytes: Image size
        - provider: DSL provider used
    """
    from .tools.render import render_layout as _render

    return _render(layout=layout, format=format, provider=provider)


@mcp.tool
def search_layouts(
    query: str,
    k: int = 5,
    source_filter: str | None = None,
) -> dict[str, Any]:
    """Search for similar layouts in the vector database.

    Finds layouts semantically similar to the query. Useful for
    inspiration or understanding existing patterns.

    Args:
        query: Natural language description of desired layout.
        k: Number of results (1-20). Default: 5
        source_filter: Filter by corpus source (optional).

    Returns:
        Dictionary with:
        - results: List of similar layouts with scores
        - total_in_index: Total items in index
        - query: Original query
    """
    from .tools.search import search_layouts as _search

    return _search(query=query, k=k, source_filter=source_filter)


# =============================================================================
# Resources
# =============================================================================


@mcp.resource("schema://components")
def get_component_schema() -> str:
    """Get the component type catalog.

    Returns JSON schema describing all 26 UI component types
    with their categories, constraints, and metadata.
    """
    from src.schema import export_llm_schema

    return export_llm_schema()


@mcp.resource("schema://layout")
def get_layout_schema() -> str:
    """Get the LayoutNode JSON schema.

    Returns the full JSON schema for LayoutNode structures,
    including all fields, types, and constraints.
    """
    import json

    from src.mid import LayoutNode

    schema = LayoutNode.model_json_schema()
    return json.dumps(schema, indent=2)


@mcp.resource("config://models")
def get_available_models() -> str:
    """Get list of available LLM models.

    Returns information about LLM models that can be used
    with generate_layout, including which are currently available.
    """
    import json

    from src.config import get_available_llm_providers

    available_providers = get_available_llm_providers()

    try:
        from src.llm import LLMModel

        models = []
        for model in LLMModel:
            models.append(
                {
                    "name": model.spec.name,
                    "provider": model.spec.provider,
                    "available": model.spec.provider in available_providers,
                }
            )
        return json.dumps(
            {
                "models": models,
                "available_providers": available_providers,
            }
        )
    except ImportError:
        return json.dumps({"available_providers": available_providers})


@mcp.resource("config://providers")
def get_available_providers() -> str:
    """Get list of available DSL providers.

    Returns information about transpilation providers (D2, PlantUML)
    and corpus providers (Rico, Enrico, etc.).
    """
    import json

    from src.providers import list_providers

    dsl_providers = list_providers()

    try:
        from src.corpus.api import CorpusManager

        corpus_providers = CorpusManager().list_providers()
    except ImportError:
        corpus_providers = []

    return json.dumps(
        {
            "dsl_providers": dsl_providers,
            "corpus_providers": corpus_providers,
        }
    )


# =============================================================================
# Server Factory
# =============================================================================


def create_server() -> FastMCP:
    """Create and configure the MCP server instance.

    Returns:
        Configured FastMCP server instance.
    """
    return mcp


# =============================================================================
# Server Runner
# =============================================================================


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
        stream=sys.stderr,  # MCP uses stdout for protocol, log to stderr
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
