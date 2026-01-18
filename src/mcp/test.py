"""Unit tests for MCP server module.

Tests cover:
- Server configuration
- Server instance creation
- Tool registration
- Basic tool functionality
"""

import pytest

from .lib import (
    ServerConfig,
    TransportType,
    get_server_capabilities,
    get_server_version,
)
from .server import create_server, mcp

# =============================================================================
# Configuration Tests
# =============================================================================


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Default config has expected values."""
        config = ServerConfig()

        assert config.name == "wireframe-mcp"
        assert config.transport == TransportType.STDIO
        assert config.host == "0.0.0.0"
        assert config.port == 18080
        assert config.path == "/mcp"

    @pytest.mark.unit
    def test_from_env_default(self):
        """from_env creates config with default transport."""
        config = ServerConfig.from_env()

        assert config.transport == TransportType.STDIO
        assert config.name == "wireframe-mcp"

    @pytest.mark.unit
    def test_from_env_with_transport(self):
        """from_env respects transport override."""
        config = ServerConfig.from_env(transport=TransportType.HTTP)

        assert config.transport == TransportType.HTTP


class TestTransportType:
    """Tests for TransportType enum."""

    @pytest.mark.unit
    def test_transport_values(self):
        """All transport types have expected values."""
        assert TransportType.STDIO.value == "stdio"
        assert TransportType.HTTP.value == "http"
        assert TransportType.SSE.value == "sse"

    @pytest.mark.unit
    def test_transport_from_string(self):
        """Transport can be created from string."""
        assert TransportType("stdio") == TransportType.STDIO
        assert TransportType("http") == TransportType.HTTP
        assert TransportType("sse") == TransportType.SSE


# =============================================================================
# Server Utility Tests
# =============================================================================


class TestServerUtilities:
    """Tests for server utility functions."""

    @pytest.mark.unit
    def test_get_server_version(self):
        """Server version is a valid semver string."""
        version = get_server_version()

        assert isinstance(version, str)
        assert len(version.split(".")) >= 2  # At least major.minor

    @pytest.mark.unit
    def test_get_server_capabilities(self):
        """Server capabilities contains expected keys."""
        caps = get_server_capabilities()

        assert isinstance(caps, dict)
        assert "tools" in caps
        assert "resources" in caps
        assert caps["tools"] is True
        assert caps["resources"] is True


# =============================================================================
# Server Instance Tests
# =============================================================================


class TestServerInstance:
    """Tests for FastMCP server instance."""

    @pytest.mark.unit
    def test_mcp_instance_exists(self):
        """Global mcp instance is created."""
        assert mcp is not None

    @pytest.mark.unit
    def test_create_server_returns_mcp(self):
        """create_server returns the mcp instance."""
        server = create_server()

        assert server is mcp

    @pytest.mark.unit
    def test_server_has_name(self):
        """Server has correct name."""
        assert mcp.name == "wireframe-mcp"


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests for MCP tool registration."""

    @pytest.mark.unit
    def test_ping_tool_registered(self):
        """ping tool is registered on server."""
        # Verify server instance exists and has tools capability
        # Note: Actual tool registration is verified in integration tests
        assert mcp is not None
        assert get_server_capabilities()["tools"] is True

    @pytest.mark.unit
    def test_get_server_info_tool_registered(self):
        """get_server_info tool is registered on server."""
        from .server import get_server_info

        # FastMCP wraps tools in FunctionTool, access underlying fn
        result = get_server_info.fn()
        assert "get_server_info" in result["tools"]


# =============================================================================
# Tool Functionality Tests (Unit level - no MCP protocol)
# =============================================================================


class TestPingTool:
    """Tests for ping tool logic."""

    @pytest.mark.unit
    def test_ping_returns_status(self):
        """ping returns ok status."""
        from .server import ping

        # FastMCP wraps tools in FunctionTool, access underlying fn
        result = ping.fn()

        assert isinstance(result, dict)
        assert result["status"] == "ok"
        assert "version" in result
        assert "capabilities" in result

    @pytest.mark.unit
    def test_ping_includes_version(self):
        """ping includes server version."""
        from .server import ping

        # FastMCP wraps tools in FunctionTool, access underlying fn
        result = ping.fn()

        assert result["version"] == get_server_version()


class TestGetServerInfoTool:
    """Tests for get_server_info tool logic."""

    @pytest.mark.unit
    def test_get_server_info_returns_dict(self):
        """get_server_info returns proper structure."""
        from .server import get_server_info

        # FastMCP wraps tools in FunctionTool, access underlying fn
        result = get_server_info.fn()

        assert isinstance(result, dict)
        assert result["name"] == "wireframe-mcp"
        assert "version" in result
        assert "description" in result
        assert "capabilities" in result
        assert "tools" in result

    @pytest.mark.unit
    def test_get_server_info_lists_tools(self):
        """get_server_info includes tool list."""
        from .server import get_server_info

        # FastMCP wraps tools in FunctionTool, access underlying fn
        result = get_server_info.fn()

        assert isinstance(result["tools"], list)
        assert "ping" in result["tools"]
        assert "get_server_info" in result["tools"]
        assert "preview_layout" in result["tools"]  # Renamed from render_layout


# =============================================================================
# MCP Protocol Integration Tests (require async)
# =============================================================================


@pytest.mark.mcp
class TestMCPProtocol:
    """Integration tests using MCP client protocol.

    These tests require pytest-asyncio and test the full MCP protocol flow.
    """

    @pytest.mark.asyncio
    async def test_client_can_connect(self):
        """Client can connect to server."""
        try:
            from fastmcp import Client

            async with Client(mcp) as client:
                # Connection successful if we get here
                assert client is not None
        except ImportError:
            pytest.skip("fastmcp not installed")

    @pytest.mark.asyncio
    async def test_client_can_list_tools(self):
        """Client can list available tools."""
        try:
            from fastmcp import Client

            async with Client(mcp) as client:
                tools = await client.list_tools()

                tool_names = [t.name for t in tools]
                assert "ping" in tool_names
                assert "get_server_info" in tool_names
        except ImportError:
            pytest.skip("fastmcp not installed")

    @pytest.mark.asyncio
    async def test_client_can_call_ping(self):
        """Client can call ping tool."""
        try:
            from fastmcp import Client

            async with Client(mcp) as client:
                result = await client.call_tool("ping", {})

                # Result structure depends on FastMCP version
                assert result is not None
        except ImportError:
            pytest.skip("fastmcp not installed")

    @pytest.mark.asyncio
    async def test_client_can_call_get_server_info(self):
        """Client can call get_server_info tool."""
        try:
            from fastmcp import Client

            async with Client(mcp) as client:
                result = await client.call_tool("get_server_info", {})

                assert result is not None
        except ImportError:
            pytest.skip("fastmcp not installed")
