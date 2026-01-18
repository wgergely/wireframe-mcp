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

from fastmcp import FastMCP

from .lib import (
    ServerConfig,
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
# Health Check Tool (for testing server connectivity)
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
            # Phase 2 tools (to be implemented):
            # "generate_layout",
            # "search_layouts",
            # "render_layout",
            # "validate_layout",
            # "transpile_layout",
        ],
    }


# =============================================================================
# Server Factory
# =============================================================================


def create_server(config: ServerConfig | None = None) -> FastMCP:
    """Create and configure the MCP server instance.

    Args:
        config: Optional server configuration.

    Returns:
        Configured FastMCP server instance.
    """
    # The global mcp instance is already configured
    # This function exists for testing and custom configuration
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
