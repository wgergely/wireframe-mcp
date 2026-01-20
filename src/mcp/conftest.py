"""Pytest fixtures for MCP server tests.

This module provides:
- FASTMCP_AVAILABLE check for graceful degradation
- Automatic skipping of MCP tests when fastmcp not installed
- Server and client fixtures for protocol testing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

import pytest

# Check if fastmcp is available
try:
    from fastmcp import Client

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

if TYPE_CHECKING:
    from fastmcp import FastMCP


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Skip MCP tests if fastmcp not installed.

    Automatically adds skip marker to all tests with @pytest.mark.mcp
    when the fastmcp package is not available.
    """
    if not FASTMCP_AVAILABLE:
        skip_mcp = pytest.mark.skip(reason="fastmcp not installed")
        for item in items:
            # Check for the actual pytest marker, not just keyword
            # (package name 'mcp' is also added to keywords by pytest)
            if item.get_closest_marker("mcp") is not None:
                item.add_marker(skip_mcp)


# =============================================================================
# Server Fixtures
# =============================================================================


@pytest.fixture
def mcp_server() -> FastMCP:
    """Create MCP server instance for testing.

    Returns:
        Configured FastMCP server instance.

    Note:
        Skips test if fastmcp is not installed.
    """
    if not FASTMCP_AVAILABLE:
        pytest.skip("fastmcp not installed")

    try:
        from .server import create_server
    except ImportError:
        pytest.skip("fastmcp not installed")

    return create_server()


@pytest.fixture
async def mcp_client(mcp_server: FastMCP) -> AsyncGenerator[Client, None]:
    """Create connected MCP client for testing.

    Args:
        mcp_server: The MCP server instance.

    Yields:
        Connected Client instance for testing.

    Note:
        Skips test if fastmcp is not installed.
    """
    if not FASTMCP_AVAILABLE:
        pytest.skip("fastmcp not installed")

    from fastmcp import Client

    async with Client(mcp_server) as client:
        yield client
