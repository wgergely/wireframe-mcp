"""Docker configuration and utilities for wireframe-mcp.

This module provides constants and utility functions for Docker container
configuration, including image names, registry settings, and container paths.
"""

from pathlib import PurePosixPath

__all__ = [
    "DOCKER_IMAGE_NAME",
    "DOCKER_REGISTRY",
    "CORPUS_DATA_PATH",
    "CONFIG_PATH",
    "get_container_path",
]

# Docker image configuration
DOCKER_IMAGE_NAME: str = "wireframe-mcp"
DOCKER_REGISTRY: str = "ghcr.io/wgergely"

# Container paths (POSIX paths for Linux containers)
CORPUS_DATA_PATH: PurePosixPath = PurePosixPath("/app/corpus")
CONFIG_PATH: PurePosixPath = PurePosixPath("/app/config")


def get_container_path(subpath: str, base: PurePosixPath = CORPUS_DATA_PATH) -> str:
    """Construct a container path from a subpath.

    Args:
        subpath: Relative path within the container.
        base: Base path in container. Defaults to CORPUS_DATA_PATH.

    Returns:
        Full container path as string.
    """
    return str(base / subpath)
