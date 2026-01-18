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

Available Tools (Phase 1):
    - ping: Health check
    - get_server_info: Server metadata

Planned Tools (Phase 2):
    - generate_layout: Generate UI layouts from natural language
    - search_layouts: Search vector database for similar layouts
    - render_layout: Render layouts to PNG/SVG
    - validate_layout: Validate layout structure
    - transpile_layout: Convert layouts to DSL code
"""

from .lib import (
    ServerConfig,
    TransportType,
    get_server_capabilities,
    get_server_version,
)
from .server import create_server, mcp, run_server

__all__ = [
    # Server instance
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
