"""Docker module for wireframe-mcp containerization."""

from docker.lib import (
    CONFIG_PATH,
    CORPUS_DATA_PATH,
    DOCKER_IMAGE_NAME,
    DOCKER_REGISTRY,
)

__all__ = [
    "DOCKER_IMAGE_NAME",
    "DOCKER_REGISTRY",
    "CORPUS_DATA_PATH",
    "CONFIG_PATH",
]
