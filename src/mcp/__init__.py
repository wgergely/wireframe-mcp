"""MCP (Model Context Protocol) server for wireframe-mcp.

This module provides the MCP server implementation that exposes
UI layout generation capabilities to LLM clients like Claude Desktop.

Example:
    # Start server in STDIO mode (for Claude Desktop)
    >>> from src.mcp.server import run_server
    >>> run_server()

    # Start server in HTTP mode
    >>> from src.mcp.server import run_server, TransportType
    >>> run_server(transport=TransportType.HTTP, port=18080)

    # Create server for testing
    >>> from src.mcp.server import create_server
    >>> server = create_server()

Available Tools:
    - ping: Health check
    - get_server_info: Server metadata
    - generate_layout: Generate UI layouts from natural language (returns draft + JSON)
    - search_layouts: Search vector database for similar layouts
    - preview_layout: Render layouts to PNG/SVG wireframe images
    - validate_layout: Validate layout structure
"""

from .lib import (
    ServerConfig,
    TransportType,
    get_server_capabilities,
    get_server_version,
)

# Conditionally import server module (requires fastmcp)
try:
    from .server import create_server, mcp, run_server

    _FASTMCP_AVAILABLE = True
except ImportError:
    # fastmcp not installed - server features unavailable
    create_server = None  # type: ignore[assignment,misc]
    mcp = None  # type: ignore[assignment]
    run_server = None  # type: ignore[assignment]
    _FASTMCP_AVAILABLE = False

__all__ = [
    # Server instance (requires fastmcp)
    "mcp",
    "create_server",
    "run_server",
    # Configuration
    "ServerConfig",
    "TransportType",
    # Utilities
    "get_server_version",
    "get_server_capabilities",
]
