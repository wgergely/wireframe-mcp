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

# Conditionally import server module (requires fastmcp)
try:
    from .server import create_server, mcp

    FASTMCP_AVAILABLE = True
except ImportError:
    create_server = None  # type: ignore[assignment,misc]
    mcp = None  # type: ignore[assignment]
    FASTMCP_AVAILABLE = False

# Skip marker for tests requiring fastmcp
requires_fastmcp = pytest.mark.skipif(
    not FASTMCP_AVAILABLE,
    reason="fastmcp not installed",
)

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


@requires_fastmcp
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


@requires_fastmcp
class TestToolRegistration:
    """Tests for MCP tool registration."""

    @pytest.mark.unit
    def test_get_artifact_tool_registered(self):
        """get_artifact tool is registered for variations workflow."""
        tool_names = set(mcp._tool_manager._tools.keys())

        assert "get_artifact" in tool_names

    @pytest.mark.unit
    def test_core_tools_registered(self):
        """Core LLM tools are registered on server."""
        tool_names = set(mcp._tool_manager._tools.keys())

        # Core user workflow tools
        assert "generate_layout" in tool_names
        assert "preview_layout" in tool_names
        assert "generate_variations" in tool_names

        # Health check
        assert "status" in tool_names

    @pytest.mark.unit
    def test_only_essential_tools_exposed(self):
        """Only essential tools are exposed (no dev/admin tools)."""
        tool_names = set(mcp._tool_manager._tools.keys())

        # Should have exactly 7 tools:
        # - Core: generate_layout, preview_layout, generate_variations, get_artifact
        # - Status: status, help, list_models
        assert len(tool_names) == 7

        # Verify expected tools
        assert "help" in tool_names
        assert "list_models" in tool_names

        # These should NOT be exposed (dev/admin concerns)
        assert "get_server_info" not in tool_names
        assert "validate_layout" not in tool_names
        assert "search_layouts" not in tool_names
        assert "get_history" not in tool_names
        assert "get_sessions" not in tool_names
        assert "get_storage_stats" not in tool_names


# =============================================================================
# Tool Functionality Tests (Unit level - no MCP protocol)
# =============================================================================


@requires_fastmcp
class TestStatusTool:
    """Tests for status tool logic."""

    @pytest.mark.unit
    def test_status_returns_health(self):
        """status returns health status."""
        from .server import status

        result = status.fn()

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in result

    @pytest.mark.unit
    def test_status_includes_capabilities(self):
        """status includes capability information."""
        from .server import status

        result = status.fn()

        assert "capabilities" in result
        assert "generate_layout" in result["capabilities"]
        assert "preview_layout" in result["capabilities"]

    @pytest.mark.unit
    def test_status_includes_services(self):
        """status includes service status."""
        from .server import status

        result = status.fn()

        assert "services" in result
        assert "kroki" in result["services"]
        assert "rag_index" in result["services"]
        assert "llm_providers" in result["services"]


# =============================================================================
# Validation Tests
# =============================================================================


@requires_fastmcp
class TestValidationHelpers:
    """Tests for parameter validation functions."""

    @pytest.mark.unit
    def test_validate_temperature_valid(self):
        """Valid temperatures don't raise."""
        from .server import _validate_temperature

        _validate_temperature(0.0)
        _validate_temperature(1.0)
        _validate_temperature(2.0)
        _validate_temperature(0.7)

    @pytest.mark.unit
    def test_validate_temperature_invalid(self):
        """Invalid temperatures raise ValueError."""
        from .server import _validate_temperature

        with pytest.raises(ValueError, match="Temperature must be"):
            _validate_temperature(-0.1)
        with pytest.raises(ValueError, match="Temperature must be"):
            _validate_temperature(2.1)

    @pytest.mark.unit
    def test_validate_provider_valid(self):
        """Valid providers don't raise."""
        from .server import _validate_provider

        _validate_provider("d2")
        _validate_provider("plantuml")

    @pytest.mark.unit
    def test_validate_provider_invalid(self):
        """Invalid providers raise ValueError."""
        from .server import _validate_provider

        with pytest.raises(ValueError, match="Invalid provider"):
            _validate_provider("unknown")
        with pytest.raises(ValueError, match="Invalid provider"):
            _validate_provider("mermaid")


# =============================================================================
# MCP Protocol Integration Tests (require async)
# =============================================================================


@pytest.mark.mcp
class TestMCPProtocol:
    """Integration tests using MCP client protocol.

    These tests require pytest-asyncio and use the mcp_client fixture
    which handles fastmcp availability checking automatically.
    """

    @pytest.mark.asyncio
    async def test_client_can_connect(self, mcp_client):
        """Client can connect to server."""
        assert mcp_client is not None

    @pytest.mark.asyncio
    async def test_client_can_list_tools(self, mcp_client):
        """Client can list available tools."""
        tools = await mcp_client.list_tools()

        tool_names = [t.name for t in tools]
        assert "status" in tool_names
        assert "generate_layout" in tool_names
        assert "preview_layout" in tool_names
        assert "generate_variations" in tool_names

    @pytest.mark.asyncio
    async def test_client_can_call_status(self, mcp_client):
        """Client can call status tool."""
        result = await mcp_client.call_tool("status", {})
        assert result is not None

    @pytest.mark.asyncio
    async def test_client_can_call_get_artifact_not_found(self, mcp_client):
        """get_artifact returns error for unknown ID."""
        # Tool should handle error gracefully
        with pytest.raises(Exception, match="not found"):
            await mcp_client.call_tool(
                "get_artifact", {"artifact_id": "nonexistent-id-12345"}
            )
