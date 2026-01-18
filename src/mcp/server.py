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

# Valid output formats for rendering
VALID_FORMATS = frozenset(("png", "svg"))


# =============================================================================
# Validation Helpers
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


def _validate_format(output_format: str) -> None:
    """Validate output format parameter."""
    if output_format not in VALID_FORMATS:
        raise ValueError(
            f"Invalid format '{output_format}'. Valid: {sorted(VALID_FORMATS)}"
        )


def _validate_temperature(temperature: float) -> None:
    """Validate temperature parameter."""
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"Temperature must be 0.0-2.0, got {temperature}")


def _validate_k(k: int) -> None:
    """Validate k (number of results) parameter."""
    if not 1 <= k <= 20:
        raise ValueError(f"k must be 1-20, got {k}")


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
            "generate_variations",
            "validate_layout",
            "preview_layout",
            "search_layouts",
            "get_history",
            "get_artifact",
            "get_storage_stats",
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
    persist: bool = True,
    session_id: str | None = None,
    parent_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a UI layout from natural language description.

    This is the primary creation tool. It returns structured JSON and a
    draft text tree for quick human review. Use preview_layout to get
    a visual wireframe image.

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
        persist: Save to history for later retrieval. Default: True
        session_id: Attach to specific session (optional).
        parent_id: Link to parent artifact for refinements (optional).
        tags: User-defined tags for organization (optional).

    Returns:
        Dictionary with:
        - artifact_id: UUID of stored artifact (if persist=True)
        - layout: Generated LayoutNode as JSON
        - draft: Human-readable text tree for quick review
        - stats: Generation statistics
    """
    _validate_temperature(temperature)
    _validate_provider(provider)

    from .tools.generate import generate_layout as _generate

    return _generate(
        query=query,
        model=model,
        temperature=temperature,
        provider=provider,
        include_rag=include_rag,
        persist=persist,
        session_id=session_id,
        parent_id=parent_id,
        tags=tags,
    )


@mcp.tool
def generate_variations(
    query: str,
    count: int = 3,
    temperature_range: tuple[float, float] | None = None,
    model: str | None = None,
    include_rag: bool = True,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Generate multiple layout variations for comparison.

    Creates N different layouts from the same query with varying
    temperatures to produce diversity. Useful for exploring alternatives.

    Args:
        query: Natural language description of the desired layout.
        count: Number of variations (1-10). Default: 3
        temperature_range: Optional (min, max) temperature spread.
        model: LLM model to use (optional).
        include_rag: Include RAG context. Default: True
        session_id: Attach to specific session (optional).

    Returns:
        Dictionary with:
        - variation_set_id: UUID of the variation set
        - artifacts: List of artifact summaries
        - diversity_score: How different the variations are (0-1)
        - rankings: Artifacts ranked by different criteria
    """
    from .tools.generate import generate_variations as _generate_variations

    return _generate_variations(
        query=query,
        count=count,
        temperature_range=temperature_range,
        model=model,
        include_rag=include_rag,
        session_id=session_id,
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
def preview_layout(
    layout: dict[str, Any],
    style: str = "wireframe",
    output_format: str = "png",
) -> dict[str, Any]:
    """Render a layout to a visual wireframe image.

    Use this tool to see a visual preview of your layout.
    Requires Kroki service to be running.

    Args:
        layout: Layout JSON to render.
        style: Visual style. Options:
            - "wireframe": Clean UI mockup (default)
            - "sketch": Hand-drawn appearance
            - "minimal": Simple boxes
        output_format: Output format ("png", "svg"). Default: "png"

    Returns:
        Dictionary with:
        - image_data: Base64-encoded image
        - format: Image format
        - style: Visual style used
        - size_bytes: Image size
    """
    _validate_format(output_format)

    from .tools.preview import preview_layout as _preview

    return _preview(layout=layout, style=style, output_format=output_format)


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
    _validate_k(k)

    from .tools.search import search_layouts as _search

    return _search(query=query, k=k, source_filter=source_filter)


# =============================================================================
# History Tools
# =============================================================================


@mcp.tool
def get_history(
    session_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    query: str | None = None,
) -> dict[str, Any]:
    """Retrieve generation history.

    Lists past generations with optional filtering.

    Args:
        session_id: Filter by session ID (optional).
        limit: Maximum results (1-100). Default: 20
        offset: Number of results to skip. Default: 0
        query: Semantic search query (optional).

    Returns:
        Dictionary with:
        - artifacts: List of artifact summaries
        - total_count: Total matching artifacts
        - has_more: Whether more results exist
    """
    from .tools.history import get_history as _get_history

    return _get_history(
        session_id=session_id,
        limit=limit,
        offset=offset,
        query=query,
    )


@mcp.tool
def get_artifact(
    artifact_id: str,
    include_lineage: bool = False,
) -> dict[str, Any]:
    """Retrieve a specific artifact by ID.

    Args:
        artifact_id: The artifact UUID.
        include_lineage: Include parent/child relationships. Default: False

    Returns:
        Dictionary with full artifact data including layout and stats.
    """
    from .tools.history import get_artifact as _get_artifact

    return _get_artifact(artifact_id=artifact_id, include_lineage=include_lineage)


@mcp.tool
def get_storage_stats() -> dict[str, Any]:
    """Get history storage statistics.

    Returns:
        Dictionary with storage stats:
        - total_size_mb: Total storage size
        - artifact_count: Number of artifacts
        - session_count: Number of sessions
        - oldest_artifact_days: Age of oldest artifact
    """
    from .tools.history import get_storage_stats as _get_storage_stats

    return _get_storage_stats()


# =============================================================================
# Resources (with caching for performance)
# =============================================================================


@lru_cache(maxsize=1)
def _cached_component_schema() -> str:
    """Cached component schema - rarely changes at runtime."""
    from src.schema import export_llm_schema

    return export_llm_schema()


@lru_cache(maxsize=1)
def _cached_layout_schema() -> str:
    """Cached layout schema - rarely changes at runtime."""
    from src.mid import LayoutNode

    schema = LayoutNode.model_json_schema()
    return json.dumps(schema, indent=2)


@lru_cache(maxsize=1)
def _cached_providers() -> str:
    """Cached provider list - rarely changes at runtime."""
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


@mcp.resource("schema://components")
def get_component_schema() -> str:
    """Get the component type catalog.

    Returns JSON schema describing all 26 UI component types
    with their categories, constraints, and metadata.
    """
    return _cached_component_schema()


@mcp.resource("schema://layout")
def get_layout_schema() -> str:
    """Get the LayoutNode JSON schema.

    Returns the full JSON schema for LayoutNode structures,
    including all fields, types, and constraints.
    """
    return _cached_layout_schema()


@mcp.resource("config://models")
def get_available_models() -> str:
    """Get list of available LLM models.

    Returns information about LLM models that can be used
    with generate_layout, including which are currently available.

    Note: Not cached as availability may change based on env vars.
    """
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
    return _cached_providers()


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
