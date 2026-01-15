"""Docker configuration and utilities for wireframe-mcp.

This module provides constants and utility functions for Docker container
configuration, including image names, registry settings, volume names,
and container paths.
"""

from pathlib import Path, PurePosixPath

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
    "get_container_path",
    "get_compose_files",
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

# Kroki renderer configuration (internal network)
KROKI_HOST: str = "kroki"
KROKI_PORT: int = 8000


def get_container_path(subpath: str, base: PurePosixPath = CORPUS_DATA_PATH) -> str:
    """Construct a container path from a subpath.

    Args:
        subpath: Relative path within the container.
        base: Base path in container. Defaults to CORPUS_DATA_PATH.

    Returns:
        Full container path as string.
    """
    return str(base / subpath)


def get_compose_files(mode: str = "dev") -> list[Path]:
    """Get the docker-compose files for a given mode.

    Args:
        mode: Either 'dev' or 'prod'. Defaults to 'dev'.

    Returns:
        List of Path objects for the compose files.

    Raises:
        ValueError: If mode is not 'dev' or 'prod'.
    """
    if mode not in ("dev", "prod"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'dev' or 'prod'.")

    docker_dir = Path(__file__).parent
    base_file = docker_dir / "docker-compose.yml"
    override_file = docker_dir / f"docker-compose.{mode}.yml"

    return [base_file, override_file]
