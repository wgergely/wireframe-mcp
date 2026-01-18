"""Core MCP server logic for wireframe-mcp.

Provides factory functions and configuration for creating MCP server instances.
"""

from dataclasses import dataclass
from enum import Enum

from src.config import EnvVar, get_environment


class TransportType(str, Enum):
    """Supported MCP transport types."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass
class ServerConfig:
    """Configuration for MCP server.

    Attributes:
        name: Server display name.
        transport: Transport type for communication.
        host: Bind address for HTTP/SSE transports.
        port: Port for HTTP/SSE transports.
        path: URL path for HTTP transport.
    """

    name: str = "wireframe-mcp"
    transport: TransportType = TransportType.STDIO
    host: str = "0.0.0.0"
    port: int = 18080
    path: str = "/mcp"

    @classmethod
    def from_env(
        cls,
        transport: TransportType | None = None,
    ) -> "ServerConfig":
        """Create config from environment variables.

        Args:
            transport: Override transport type (default: STDIO).

        Returns:
            ServerConfig with values from environment.
        """
        return cls(
            name="wireframe-mcp",
            transport=transport or TransportType.STDIO,
            host=get_environment(EnvVar.MCP_HOST),
            port=get_environment(EnvVar.MCP_PORT),
        )


def get_server_version() -> str:
    """Get server version string."""
    return "0.1.0"


def get_server_capabilities() -> dict:
    """Get server capabilities for MCP protocol.

    Returns:
        Dictionary of capability flags.
    """
    return {
        "tools": True,
        "resources": True,
        "prompts": False,  # Not implemented yet
        "logging": True,
    }


__all__ = [
    "TransportType",
    "ServerConfig",
    "get_server_version",
    "get_server_capabilities",
]
