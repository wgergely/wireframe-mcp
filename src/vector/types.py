"""Centralized type definitions and constants for the vector module.

This module provides type-safe definitions for metadata, configuration,
and constants used throughout the vector indexing system.
"""

import os
from enum import Enum
from typing import TypedDict


def get_default_workers() -> int:
    """Get default worker count based on available CPU cores.

    Returns:
        Number of CPU cores, or 4 as fallback if detection fails.
    """
    return os.cpu_count() or 4


class BackendType(str, Enum):
    """Supported embedding backend types."""

    VOYAGE = "voyage"
    LOCAL = "local"


class OrientationCode(str, Enum):
    """Short codes for layout orientation in linearized format."""

    HORIZONTAL = "H"
    VERTICAL = "V"
    OVERLAY = "O"


# Mapping from orientation string values to short codes
ORIENTATION_CODES: dict[str, str] = {
    "horizontal": OrientationCode.HORIZONTAL.value,
    "vertical": OrientationCode.VERTICAL.value,
    "overlay": OrientationCode.OVERLAY.value,
}

# Default orientation value (from LayoutNode)
DEFAULT_ORIENTATION = "vertical"


class ItemMetadata(TypedDict):
    """Type-safe metadata for indexed items."""

    source: str
    dataset: str
    text: str
    node_count: int
    max_depth: int
    component_summary: dict[str, int]


class MetadataKey(str, Enum):
    """Keys for item metadata dictionary."""

    SOURCE = "source"
    DATASET = "dataset"
    TEXT = "text"
    NODE_COUNT = "node_count"
    MAX_DEPTH = "max_depth"
    COMPONENT_SUMMARY = "component_summary"


# Default values for missing metadata
UNKNOWN_SOURCE = "unknown"
UNKNOWN_DATASET = "unknown"


class EnvVar(str, Enum):
    """Environment variable names for vector configuration."""

    GPU_ENABLED = "VECTOR_USE_GPU"


class BoolStr(str, Enum):
    """Boolean string values for environment variables."""

    TRUE = "true"
    FALSE = "false"


def parse_bool_env(value: str) -> bool | None:
    """Parse boolean environment variable.

    Args:
        value: Environment variable value.

    Returns:
        True if "true", False if "false", None otherwise.
    """
    value = value.lower().strip()
    if value == BoolStr.TRUE.value:
        return True
    elif value == BoolStr.FALSE.value:
        return False
    return None


# FAISS configuration
OPTIMAL_ADD_BATCH_SIZE = 10000

# Default batch sizes
DEFAULT_EMBEDDING_BATCH_SIZE = 100
# Note: DEFAULT_SERIALIZATION_WORKERS is computed dynamically via get_default_workers()
# to use available CPU cores. For backward compatibility, we keep this constant
# but it should be replaced with get_default_workers() calls.
DEFAULT_SERIALIZATION_WORKERS = get_default_workers()

# Voyage API configuration
VOYAGE_BATCH_SIZE = 128
# Network I/O concurrent batches: kept low to avoid rate limits.
# Unlike CPU-bound tasks, more concurrent requests don't always help
# and can trigger API throttling. 3 is a safe default for most APIs.
VOYAGE_CONCURRENT_BATCHES = 3
VOYAGE_RATE_LIMIT_FAST_THRESHOLD = 0.1
VOYAGE_RATE_LIMIT_MODERATE_THRESHOLD = 0.3
VOYAGE_RATE_LIMIT_MIN_DELAY = 0.02
VOYAGE_RATE_LIMIT_MODERATE_DELAY = 0.05
VOYAGE_RETRY_BASE = 2  # For exponential backoff: RETRY_BASE ** attempt


__all__ = [
    # Enums
    "BackendType",
    "OrientationCode",
    "MetadataKey",
    "EnvVar",
    "BoolStr",
    # TypedDict
    "ItemMetadata",
    # Constants
    "ORIENTATION_CODES",
    "DEFAULT_ORIENTATION",
    "UNKNOWN_SOURCE",
    "UNKNOWN_DATASET",
    "OPTIMAL_ADD_BATCH_SIZE",
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_SERIALIZATION_WORKERS",
    "VOYAGE_BATCH_SIZE",
    "VOYAGE_CONCURRENT_BATCHES",
    "VOYAGE_RATE_LIMIT_FAST_THRESHOLD",
    "VOYAGE_RATE_LIMIT_MODERATE_THRESHOLD",
    "VOYAGE_RATE_LIMIT_MIN_DELAY",
    "VOYAGE_RATE_LIMIT_MODERATE_DELAY",
    "VOYAGE_RETRY_BASE",
    # Functions
    "parse_bool_env",
    "get_default_workers",
]
