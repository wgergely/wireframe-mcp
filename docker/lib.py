"""Docker configuration and utilities for wireframe-mcp.

This module provides constants and utility functions for Docker container
configuration, including image names, registry settings, volume names,
and container paths.

Port and host configuration is managed centrally via src.config module.
"""

from pathlib import Path, PurePosixPath

from src.config import EnvVar, get_environment

__all__ = [
    "DOCKER_IMAGE_NAME",
    "DOCKER_REGISTRY",
    "CORPUS_DATA_PATH",
    "CORPUS_MODELS_PATH",
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "VOLUME_CORPUS_DATA",
    "VOLUME_CORPUS_MODELS",
    "VOLUME_OUTPUT",
    "KROKI_HOST",
    "KROKI_PORT",
    "MCP_HOST",
    "MCP_PORT",
    "get_container_path",
    "get_compose_files",
    "list_backends",
    "list_modes",
]

# Docker image configuration
DOCKER_IMAGE_NAME: str = "wireframe-mcp"
DOCKER_REGISTRY: str = "ghcr.io/wgergely"

# Named volumes
VOLUME_CORPUS_DATA: str = "wireframe-corpus-data"
VOLUME_CORPUS_MODELS: str = "wireframe-corpus-models"
VOLUME_OUTPUT: str = "wireframe-output"

# Container paths (POSIX paths for Linux containers)
CORPUS_DATA_PATH: PurePosixPath = PurePosixPath("/app/corpus/data")
CORPUS_MODELS_PATH: PurePosixPath = PurePosixPath("/app/corpus/models")
CONFIG_PATH: PurePosixPath = PurePosixPath("/app/config")
OUTPUT_PATH: PurePosixPath = PurePosixPath("/app/data")

# Kroki renderer configuration - uses centralized config
KROKI_HOST: str = "kroki"  # Docker internal hostname


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


def get_compose_files(
    mode: str = "dev",
    include_kroki: bool = False,
    include_backends: list[str] | None = None,
) -> list[Path]:
    """Get the docker-compose files for a given mode.

    Args:
        mode: Either 'dev' or 'prod'. Defaults to 'dev'.
        include_kroki: Include kroki backend. Defaults to False.
        include_backends: Additional backend compose files (e.g. ['gpu', 'postgres']).
            Defaults to None.

    Returns:
        List of Path objects for the compose files in correct order.

    Raises:
        ValueError: If mode is not 'dev' or 'prod', or invalid backend specified.
    """
    if mode not in ("dev", "prod"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'dev' or 'prod'.")

    docker_dir = Path(__file__).parent
    files = [docker_dir / "docker-compose.yml"]

    # Add mode-specific overrides
    mode_file = docker_dir / f"docker-compose.{mode}.yml"
    if mode_file.exists():
        files.append(mode_file)

    # Add kroki backend if requested
    if include_kroki:
        kroki_file = docker_dir / "docker-compose.kroki.yml"
        if kroki_file.exists():
            files.append(kroki_file)
        else:
            raise ValueError(f"Kroki compose file not found: {kroki_file}")

    # Add other backend overrides
    if include_backends:
        for backend in include_backends:
            backend_file = docker_dir / f"docker-compose.{backend}.yml"
            if backend_file.exists():
                files.append(backend_file)
            else:
                raise ValueError(
                    f"Backend compose file not found: {backend_file} "
                    f"(backend: {backend})"
                )

    return files


def list_modes() -> list[str]:
    """List available docker-compose modes.

    Returns:
        List of available modes (e.g., ['dev', 'prod']).
    """
    docker_dir = Path(__file__).parent
    modes = []
    for file in docker_dir.glob("docker-compose.*.yml"):
        # Extract mode from filename (docker-compose.MODE.yml)
        mode = file.stem.replace("docker-compose.", "")
        if mode and mode not in ("kroki",):  # Exclude special backends
            modes.append(mode)
    return sorted(modes)


def list_backends() -> list[str]:
    """List available docker-compose backends.

    Returns:
        List of available backends (e.g., ['kroki', 'gpu', 'postgres']).
    """
    docker_dir = Path(__file__).parent
    backends = []
    for file in docker_dir.glob("docker-compose.*.yml"):
        # Extract backend from filename (docker-compose.BACKEND.yml)
        name = file.stem.replace("docker-compose.", "")
        if name and name not in ("dev", "prod"):  # Exclude modes
            backends.append(name)
    return sorted(backends)
