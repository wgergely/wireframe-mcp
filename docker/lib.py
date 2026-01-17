"""Docker configuration and utilities for wireframe-mcp.

This module provides constants and utility functions for Docker container
configuration, including image names, registry settings, volume names,
and container paths.

Port and host configuration is managed centrally via src.config module.

Naming Convention:
    Project prefix: wfmcp- (wireframe-mcp abbreviated)

    Compose Files:
        compose.base.yml       - Shared networks/volumes
        compose.core.yml       - Main MCP server
        compose.renderer.yml   - Kroki rendering stack
        compose.hotreload.yml  - Dev overrides (source mounts)
        compose.production.yml - Production overrides

    Services:
        wfmcp-server           - Main MCP server
        wfmcp-kroki            - Kroki rendering gateway
        wfmcp-kroki-d2         - D2 diagram renderer
        wfmcp-kroki-mermaid    - Mermaid diagram renderer
        wfmcp-kroki-bpmn       - BPMN diagram renderer
        wfmcp-kroki-excalidraw - Excalidraw diagram renderer
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from src.config import EnvVar, get_environment

__all__ = [
    # Project configuration
    "PROJECT_NAME",
    "PROJECT_PREFIX",
    "DOCKER_IMAGE_NAME",
    "DOCKER_REGISTRY",
    "NETWORK_NAME",
    # Service enumeration
    "Service",
    "ServiceCategory",
    "ServiceInfo",
    "get_service_info",
    "list_services",
    "list_services_by_category",
    # Container paths
    "CORPUS_DATA_PATH",
    "CORPUS_MODELS_PATH",
    "CORPUS_INDEX_PATH",
    "CONFIG_PATH",
    "OUTPUT_PATH",
    # Named volumes
    "VOLUME_CORPUS_DATA",
    "VOLUME_CORPUS_MODELS",
    "VOLUME_OUTPUT",
    # Port configuration (from centralized config)
    "KROKI_HOST",
    "KROKI_PORT",
    "MCP_HOST",
    "MCP_PORT",
    # Utilities
    "get_container_path",
    "get_compose_files",
    "list_backends",
    "list_modes",
]

# =============================================================================
# Project-Level Configuration
# =============================================================================

PROJECT_NAME: str = "wireframe-mcp"
PROJECT_PREFIX: str = "wfmcp"
DOCKER_IMAGE_NAME: str = "wireframe-mcp"
DOCKER_REGISTRY: str = "ghcr.io/wgergely"
NETWORK_NAME: str = f"{PROJECT_PREFIX}-network"


# =============================================================================
# Service Definitions
# =============================================================================


class ServiceCategory(Enum):
    """Categories for Docker services.

    Categories help organize services by their role:
        core: Main application services
        renderer: Diagram rendering backends
        support: Helper services (databases, caches, etc.)
    """

    CORE = "core"
    RENDERER = "renderer"
    SUPPORT = "support"


@dataclass(frozen=True)
class ServiceInfo:
    """Metadata for a Docker service.

    Attributes:
        container_name: Full container name (e.g., wfmcp-server).
        hostname: Docker network hostname.
        image: Docker image name.
        category: Service category.
        description: Human-readable description.
        internal_port: Port inside container.
        depends_on: List of services this depends on.
    """

    container_name: str
    hostname: str
    image: str
    category: ServiceCategory
    description: str
    internal_port: int | None = None
    depends_on: tuple[str, ...] = ()


class Service(Enum):
    """All Docker services in the wireframe-mcp stack.

    Each service has associated metadata accessible via `get_service_info()`.

    Core Services:
        SERVER: Main MCP server application

    Renderer Services:
        KROKI: Kroki rendering gateway (routes to specialized renderers)
        KROKI_D2: D2 diagram renderer
        KROKI_MERMAID: Mermaid diagram renderer
        KROKI_BPMN: BPMN diagram renderer
        KROKI_EXCALIDRAW: Excalidraw whiteboard renderer
    """

    # Core services
    SERVER = ServiceInfo(
        container_name=f"{PROJECT_PREFIX}-server",
        hostname=f"{PROJECT_PREFIX}-server",
        image=f"{DOCKER_IMAGE_NAME}:latest",
        category=ServiceCategory.CORE,
        description="Main MCP server",
        internal_port=18080,
        depends_on=("KROKI",),
    )

    # Renderer services
    KROKI = ServiceInfo(
        container_name=f"{PROJECT_PREFIX}-kroki",
        hostname="kroki",
        image="yuzutech/kroki",
        category=ServiceCategory.RENDERER,
        description="Kroki rendering gateway",
        internal_port=8000,
        depends_on=("KROKI_D2", "KROKI_MERMAID", "KROKI_BPMN", "KROKI_EXCALIDRAW"),
    )
    KROKI_D2 = ServiceInfo(
        container_name=f"{PROJECT_PREFIX}-kroki-d2",
        hostname="d2",
        image="yuzutech/kroki-d2",
        category=ServiceCategory.RENDERER,
        description="D2 diagram renderer",
        internal_port=8012,
    )
    KROKI_MERMAID = ServiceInfo(
        container_name=f"{PROJECT_PREFIX}-kroki-mermaid",
        hostname="mermaid",
        image="yuzutech/kroki-mermaid",
        category=ServiceCategory.RENDERER,
        description="Mermaid diagram renderer",
        internal_port=8002,
    )
    KROKI_BPMN = ServiceInfo(
        container_name=f"{PROJECT_PREFIX}-kroki-bpmn",
        hostname="bpmn",
        image="yuzutech/kroki-bpmn",
        category=ServiceCategory.RENDERER,
        description="BPMN diagram renderer",
        internal_port=8003,
    )
    KROKI_EXCALIDRAW = ServiceInfo(
        container_name=f"{PROJECT_PREFIX}-kroki-excalidraw",
        hostname="excalidraw",
        image="yuzutech/kroki-excalidraw",
        category=ServiceCategory.RENDERER,
        description="Excalidraw whiteboard renderer",
        internal_port=8004,
    )


def get_service_info(service: Service) -> ServiceInfo:
    """Get metadata for a service.

    Args:
        service: Service enum member.

    Returns:
        ServiceInfo with container name, hostname, image, etc.
    """
    return service.value


def list_services(category: ServiceCategory | None = None) -> list[Service]:
    """List all services, optionally filtered by category.

    Args:
        category: Filter by category. None returns all services.

    Returns:
        List of Service enum members.
    """
    if category is None:
        return list(Service)
    return [s for s in Service if s.value.category == category]


def list_services_by_category() -> dict[ServiceCategory, list[Service]]:
    """Group all services by category.

    Returns:
        Dict mapping category to list of services.
    """
    result: dict[ServiceCategory, list[Service]] = {}
    for cat in ServiceCategory:
        result[cat] = list_services(cat)
    return result


# =============================================================================
# Named Volumes
# =============================================================================

VOLUME_CORPUS_DATA: str = f"{PROJECT_PREFIX}-corpus-data"
VOLUME_CORPUS_MODELS: str = f"{PROJECT_PREFIX}-corpus-models"
VOLUME_OUTPUT: str = f"{PROJECT_PREFIX}-output"


# =============================================================================
# Container Paths (POSIX paths for Linux containers)
# =============================================================================

CORPUS_DATA_PATH: PurePosixPath = PurePosixPath("/app/corpus/data")
CORPUS_MODELS_PATH: PurePosixPath = PurePosixPath("/app/corpus/models")
CORPUS_INDEX_PATH: PurePosixPath = PurePosixPath("/app/corpus/index")
CONFIG_PATH: PurePosixPath = PurePosixPath("/app/config")
OUTPUT_PATH: PurePosixPath = PurePosixPath("/app/data")


# =============================================================================
# Port Configuration (from centralized config)
# =============================================================================

# Docker internal hostname for Kroki
KROKI_HOST: str = "kroki"


def _get_kroki_port() -> int:
    """Get Kroki port from centralized config."""
    return get_environment(EnvVar.KROKI_PORT)


def _get_mcp_host() -> str:
    """Get MCP host from centralized config."""
    return get_environment(EnvVar.MCP_HOST)


def _get_mcp_port() -> int:
    """Get MCP port from centralized config."""
    return get_environment(EnvVar.MCP_PORT)


# Expose as module-level constants for backwards compatibility
# These are evaluated at import time
KROKI_PORT: int = _get_kroki_port()
MCP_HOST: str = _get_mcp_host()
MCP_PORT: int = _get_mcp_port()


def get_container_path(subpath: str, base: PurePosixPath = CORPUS_DATA_PATH) -> str:
    """Construct a container path from a subpath.

    Args:
        subpath: Relative path within the container.
        base: Base path in container. Defaults to CORPUS_DATA_PATH.

    Returns:
        Full container path as string.
    """
    return str(base / subpath)


# =============================================================================
# Compose File Management
# =============================================================================

# File categories for compose file organization
_COMPOSE_BASE = "compose.base.yml"
_COMPOSE_SERVER = "compose.core.yml"
_COMPOSE_KROKI = "compose.renderer.yml"
_COMPOSE_HOTRELOAD = "compose.hotreload.yml"
_COMPOSE_PROD = "compose.production.yml"

# Mode to overlay file mapping
_OVERRIDE_FILES = {
    "dev": _COMPOSE_HOTRELOAD,
    "hotreload": _COMPOSE_HOTRELOAD,
    "production": _COMPOSE_PROD,
}

# Files that are service definitions (not modes or base)
_SERVICE_FILES = {_COMPOSE_SERVER, _COMPOSE_KROKI}


def get_compose_files(
    mode: str = "dev",
    include_kroki: bool = True,
    include_server: bool = True,
) -> list[Path]:
    """Get the docker-compose files for a given mode.

    Compose files are returned in the correct order for docker compose:
        1. Base (network, volumes)
        2. Server service
        3. Kroki services (if requested)
        4. Mode override (hotreload or prod)

    Args:
        mode: Deployment mode - 'dev'/'hotreload' or 'prod'. Defaults to 'dev'.
        include_kroki: Include Kroki rendering services. Defaults to True.
        include_server: Include MCP server service. Defaults to True.

    Returns:
        List of Path objects for the compose files in correct order.

    Raises:
        ValueError: If mode is invalid or required files are missing.

    Example:
        >>> # Development with hot reloading
        >>> files = get_compose_files(mode="dev", include_kroki=True)

        >>> # Production deployment
        >>> files = get_compose_files(mode="production", include_kroki=True)

        >>> # Kroki-only (for testing rendering)
        >>> files = get_compose_files(include_server=False, include_kroki=True)
    """
    if mode not in _OVERRIDE_FILES:
        valid_modes = ", ".join(sorted(_OVERRIDE_FILES.keys()))
        raise ValueError(f"Invalid mode: {mode}. Must be one of: {valid_modes}")

    docker_dir = Path(__file__).parent
    files: list[Path] = []

    # 1. Base configuration (always required)
    base_file = docker_dir / _COMPOSE_BASE
    if not base_file.exists():
        raise ValueError(f"Base compose file not found: {base_file}")
    files.append(base_file)

    # 2. Server service
    if include_server:
        server_file = docker_dir / _COMPOSE_SERVER
        if not server_file.exists():
            raise ValueError(f"Server compose file not found: {server_file}")
        files.append(server_file)

    # 3. Kroki services
    if include_kroki:
        kroki_file = docker_dir / _COMPOSE_KROKI
        if not kroki_file.exists():
            raise ValueError(f"Kroki compose file not found: {kroki_file}")
        files.append(kroki_file)

    # 4. Mode overlay
    mode_filename = _OVERRIDE_FILES[mode]
    mode_file = docker_dir / mode_filename
    if mode_file.exists():
        files.append(mode_file)

    return files


def list_modes() -> list[str]:
    """List available docker-compose modes.

    Returns:
        List of available modes (e.g., ['dev', 'hotreload', 'prod']).
    """
    return sorted(_OVERRIDE_FILES.keys())


def list_backends() -> list[str]:
    """List available docker-compose backends/services.

    Returns:
        List of available backend services (e.g., ['kroki', 'server']).
    """
    docker_dir = Path(__file__).parent
    backends = []
    for file in docker_dir.glob("compose.*.yml"):
        # split compose.{name}.yml -> {name}
        parts = file.name.split(".")
        if len(parts) == 3 and parts[0] == "compose":
            name = parts[1]
            # Filter out known non-service files
            if file.name in _SERVICE_FILES:
                backends.append(name)
    return sorted(backends)
