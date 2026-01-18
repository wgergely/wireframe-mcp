"""MCP tools for wireframe-mcp.

This module provides the core MCP tools that expose layout generation,
validation, preview, and search capabilities.

Tools:
    - generate_layout: Generate UI layouts from natural language (returns draft + JSON)
    - validate_layout: Validate layout structure
    - preview_layout: Render layouts to PNG/SVG wireframe images
    - search_layouts: Search vector database for similar layouts
"""

from .generate import generate_layout
from .preview import preview_layout
from .search import search_layouts
from .validate import validate_layout

__all__ = [
    "generate_layout",
    "validate_layout",
    "preview_layout",
    "search_layouts",
]
