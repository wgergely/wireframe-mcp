"""Docker module for wireframe-mcp containerization.

This module provides Docker configuration constants and service management utilities.

Naming Convention:
    - Project prefix: wfmcp- (wireframe-mcp abbreviated)
    - Services: wfmcp-server, wfmcp-kroki, wfmcp-kroki-d2, etc.
    - Network: wfmcp-network
    - Volumes: wfmcp-corpus-data, wfmcp-corpus-models, wfmcp-output
"""

from docker.lib import (
    # Container paths
    CONFIG_PATH,
    CORPUS_DATA_PATH,
    CORPUS_INDEX_PATH,
    CORPUS_MODELS_PATH,
    # Project configuration
    DOCKER_IMAGE_NAME,
    DOCKER_REGISTRY,
    # Port configuration
    KROKI_HOST,
    KROKI_PORT,
    MCP_HOST,
    MCP_PORT,
    NETWORK_NAME,
    OUTPUT_PATH,
    PROJECT_NAME,
    PROJECT_PREFIX,
    # Named volumes
    VOLUME_CORPUS_DATA,
    VOLUME_CORPUS_MODELS,
    VOLUME_OUTPUT,
    # Service enumeration
    Service,
    ServiceCategory,
    # Utilities
    get_compose_files,
    get_container_path,
    get_service_info,
    list_backends,
    list_modes,
    list_services,
    list_services_by_category,
)

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
    # Port configuration
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
