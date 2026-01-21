"""Centralized environment configuration management for wireframe-mcp.

Provides a unified interface for all environment variables with:
- Single `get_environment()` function for all configuration
- Type-safe enum with metadata (default, type, description)
- Consistent resolution: override > environment > default

Example:
    >>> from src.config import EnvVar, get_environment
    >>>
    >>> # Get values with automatic type conversion
    >>> port = get_environment(EnvVar.KROKI_PORT)  # Returns int
    >>> api_key = get_environment(EnvVar.OPENAI_API_KEY)  # Returns str | None
    >>>
    >>> # Override at runtime
    >>> port = get_environment(EnvVar.KROKI_PORT, override=9000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, overload

# =============================================================================
# Environment Variable Configuration
# =============================================================================


@dataclass(frozen=True)
class EnvConfig:
    """Metadata for an environment variable.

    Attributes:
        name: Environment variable name (e.g., "KROKI_PORT").
        default: Default value if not set in environment.
        var_type: Python type for value conversion (str, int, bool, Path).
        description: Human-readable description.
        category: Grouping category for documentation.
    """

    name: str
    default: Any
    var_type: type
    description: str = ""
    category: str = "general"


class EnvVar(Enum):
    """All environment variables used by wireframe-mcp.

    Each member contains an EnvConfig with name, default, type, and description.
    Use with `get_environment()` for type-safe access.

    Categories:
        - llm: LLM provider API keys
        - embedding: Embedding service configuration
        - service: Service URLs, hosts, and ports
        - corpus: Data directory paths
        - docker: Docker/container configuration
    """

    # -------------------------------------------------------------------------
    # LLM API Keys
    # -------------------------------------------------------------------------
    OPENAI_API_KEY = EnvConfig(
        name="OPENAI_API_KEY",
        default=None,
        var_type=str,
        description="OpenAI API key for GPT models",
        category="llm",
    )
    ANTHROPIC_API_KEY = EnvConfig(
        name="ANTHROPIC_API_KEY",
        default=None,
        var_type=str,
        description="Anthropic API key for Claude models",
        category="llm",
    )
    DEEPSEEK_API_KEY = EnvConfig(
        name="DEEPSEEK_API_KEY",
        default=None,
        var_type=str,
        description="DeepSeek API key",
        category="llm",
    )
    QWEN_API_KEY = EnvConfig(
        name="QWEN_API_KEY",
        default=None,
        var_type=str,
        description="Qwen/Alibaba API key",
        category="llm",
    )
    LLM_PROVIDER = EnvConfig(
        name="LLM_PROVIDER",
        default=None,
        var_type=str,
        description="Preferred LLM provider (openai, anthropic, deepseek, qwen)",
        category="llm",
    )

    # -------------------------------------------------------------------------
    # Embedding Configuration
    # -------------------------------------------------------------------------
    VOYAGE_API_KEY = EnvConfig(
        name="VOYAGE_API_KEY",
        default=None,
        var_type=str,
        description="Voyage AI API key for embeddings",
        category="embedding",
    )
    EMBEDDING_BACKEND = EnvConfig(
        name="EMBEDDING_BACKEND",
        default="local",
        var_type=str,
        description="Embedding backend: 'voyage' (cloud) or 'local'",
        category="embedding",
    )
    VECTOR_USE_GPU = EnvConfig(
        name="VECTOR_USE_GPU",
        default=None,
        var_type=bool,
        description="Force GPU usage for vector ops (None=auto-detect)",
        category="embedding",
    )

    # -------------------------------------------------------------------------
    # Service Configuration (URLs and Hosts)
    # -------------------------------------------------------------------------
    KROKI_URL = EnvConfig(
        name="KROKI_URL",
        default=None,  # Computed from KROKI_PORT if not set
        var_type=str,
        description="Kroki rendering service URL",
        category="service",
    )
    OLLAMA_HOST = EnvConfig(
        name="OLLAMA_HOST",
        default="http://localhost:11434",
        var_type=str,
        description="Ollama server URL for local LLM",
        category="service",
    )
    MCP_HOST = EnvConfig(
        name="MCP_HOST",
        default="0.0.0.0",
        var_type=str,
        description="MCP server bind address",
        category="service",
    )
    MCP_PREVIEW_PROVIDER = EnvConfig(
        name="MCP_PREVIEW_PROVIDER",
        default="plantuml",
        var_type=str,
        description="Default provider for preview rendering (plantuml, d2)",
        category="service",
    )
    MCP_PREVIEW_SKETCH = EnvConfig(
        name="MCP_PREVIEW_SKETCH",
        default=False,
        var_type=bool,
        description="Enable D2 sketch/hand-drawn mode (only when provider=d2)",
        category="service",
    )

    # -------------------------------------------------------------------------
    # Service Ports (chosen to avoid common ports like 8000, 8080)
    # -------------------------------------------------------------------------
    KROKI_PORT = EnvConfig(
        name="KROKI_PORT",
        default=18000,
        var_type=int,
        description="Kroki service port (avoids 8000)",
        category="docker",
    )
    MCP_PORT = EnvConfig(
        name="MCP_PORT",
        default=18080,
        var_type=int,
        description="MCP server port (avoids 8080)",
        category="docker",
    )

    # -------------------------------------------------------------------------
    # Corpus Data Paths
    # -------------------------------------------------------------------------
    CORPUS_DATA_DIR = EnvConfig(
        name="CORPUS_DATA_DIR",
        default=None,  # Computed from repo root
        var_type=Path,
        description="Corpus data directory",
        category="corpus",
    )
    CORPUS_MODELS_DIR = EnvConfig(
        name="CORPUS_MODELS_DIR",
        default=None,  # Computed from repo root
        var_type=Path,
        description="Embedding models directory",
        category="corpus",
    )
    CORPUS_INDEX_DIR = EnvConfig(
        name="CORPUS_INDEX_DIR",
        default=None,  # Computed from repo root
        var_type=Path,
        description="Vector index directory",
        category="corpus",
    )


# =============================================================================
# Repository Root Detection
# =============================================================================


def _find_repo_root(start_path: Path | None = None) -> Path:
    """Find repository root by searching for .gitignore file.

    Args:
        start_path: Directory to start search from. Defaults to cwd.

    Returns:
        Path to repository root directory.

    Raises:
        RuntimeError: If .gitignore is not found.
    """
    current = (start_path or Path.cwd()).resolve()

    while True:
        if (current / ".gitignore").exists():
            return current

        parent = current.parent
        if parent == current:
            raise RuntimeError(
                f"Could not find repository root. No .gitignore found "
                f"starting from: {start_path or Path.cwd()}"
            )
        current = parent


# =============================================================================
# Type Conversion Helpers
# =============================================================================


def _parse_bool(value: str) -> bool | None:
    """Parse string to boolean.

    Recognizes: true/false, 1/0, yes/no (case-insensitive).
    Returns None for unrecognized values.
    """
    normalized = value.lower().strip()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    return None


def _convert_value(value: str | None, var_type: type, default: Any) -> Any:
    """Convert string value to target type.

    Args:
        value: Raw string value from environment (or None).
        var_type: Target Python type.
        default: Default value if conversion fails or value is None.

    Returns:
        Converted value or default.
    """
    if value is None:
        return default

    if var_type is str:
        return value

    if var_type is int:
        try:
            return int(value)
        except ValueError:
            return default

    if var_type is bool:
        result = _parse_bool(value)
        return result if result is not None else default

    if var_type is Path:
        return Path(value)

    # Unknown type, return as-is
    return value


# =============================================================================
# Main Interface
# =============================================================================


@overload
def get_environment(env_var: EnvVar, override: int) -> int: ...
@overload
def get_environment(env_var: EnvVar, override: str) -> str: ...
@overload
def get_environment(env_var: EnvVar, override: bool) -> bool: ...
@overload
def get_environment(env_var: EnvVar, override: Path) -> Path: ...
@overload
def get_environment(env_var: EnvVar, override: None = None) -> Any: ...


def get_environment(env_var: EnvVar, override: Any = None) -> Any:
    """Get environment variable value with type conversion.

    Resolution priority:
        1. Explicit override parameter (highest)
        2. Environment variable value
        3. Default from EnvConfig (lowest)

    Args:
        env_var: Environment variable enum member.
        override: Optional override value (bypasses env lookup).

    Returns:
        Value converted to the appropriate type (str, int, bool, or Path).

    Example:
        >>> port = get_environment(EnvVar.KROKI_PORT)
        18000
        >>> port = get_environment(EnvVar.KROKI_PORT, override=9000)
        9000
        >>> api_key = get_environment(EnvVar.OPENAI_API_KEY)
        None  # or the key if set
    """
    config: EnvConfig = env_var.value

    # Override takes highest priority
    if override is not None:
        return override

    # Check environment
    raw_value = os.environ.get(config.name)

    # Convert and return
    return _convert_value(raw_value, config.var_type, config.default)


def get_environment_info(env_var: EnvVar) -> EnvConfig:
    """Get metadata for an environment variable.

    Args:
        env_var: Environment variable enum member.

    Returns:
        EnvConfig with name, default, type, and description.
    """
    return env_var.value


# =============================================================================
# Convenience Functions (for backwards compatibility and common patterns)
# =============================================================================


def get_kroki_url(override: str | None = None) -> str:
    """Get Kroki service URL.

    Computes URL from KROKI_PORT if KROKI_URL is not set.

    Resolution: override > KROKI_URL > http://localhost:{KROKI_PORT}
    """
    if override:
        return override

    url = get_environment(EnvVar.KROKI_URL)
    if url:
        return url

    port = get_environment(EnvVar.KROKI_PORT)
    return f"http://localhost:{port}"


def get_corpus_path(
    env_var: EnvVar,
    subdir: str,
    override: Path | str | None = None,
) -> Path:
    """Get corpus directory path with fallback to repo root.

    Resolution: override > env_var > {repo_root}/.corpus/{subdir}

    Args:
        env_var: Environment variable for this path.
        subdir: Subdirectory name under .corpus.
        override: Optional path override.

    Returns:
        Resolved Path to the directory.
    """
    if override is not None:
        return Path(override)

    env_path = get_environment(env_var)
    if env_path:
        return env_path

    repo_root = _find_repo_root()
    return repo_root / ".corpus" / subdir


def get_data_dir(override: Path | str | None = None) -> Path:
    """Get corpus data directory."""
    return get_corpus_path(EnvVar.CORPUS_DATA_DIR, "data", override)


def get_models_dir(override: Path | str | None = None) -> Path:
    """Get embedding models directory."""
    return get_corpus_path(EnvVar.CORPUS_MODELS_DIR, "models", override)


def get_index_dir(override: Path | str | None = None) -> Path:
    """Get vector index directory."""
    return get_corpus_path(EnvVar.CORPUS_INDEX_DIR, "index", override)


def get_available_llm_providers() -> list[str]:
    """Get list of available LLM providers.

    Checks both cloud providers (by API key) and local providers (by availability).

    Returns:
        List of provider names (e.g., ["openai", "ollama"]).
    """
    providers = []

    # Cloud providers - check for API keys
    if get_environment(EnvVar.OPENAI_API_KEY):
        providers.append("openai")
    if get_environment(EnvVar.ANTHROPIC_API_KEY):
        providers.append("anthropic")
    if get_environment(EnvVar.DEEPSEEK_API_KEY):
        providers.append("deepseek")
    if get_environment(EnvVar.QWEN_API_KEY):
        providers.append("qwen")

    # Local providers - check if running
    ollama_url = get_environment(EnvVar.OLLAMA_HOST)
    if ollama_url:
        try:
            import httpx

            response = httpx.get(f"{ollama_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                providers.append("ollama")
        except Exception:
            pass  # Ollama not running

    return providers


def list_environment_variables(category: str | None = None) -> list[EnvVar]:
    """List all environment variables, optionally filtered by category.

    Args:
        category: Filter by category (llm, embedding, service, docker, corpus).
                 None returns all variables.

    Returns:
        List of EnvVar enum members.
    """
    if category is None:
        return list(EnvVar)

    return [var for var in EnvVar if var.value.category == category]


def get_docker_ports() -> dict[str, int]:
    """Get all Docker service ports.

    Returns:
        Dict mapping service name to port number.
    """
    return {
        "kroki": get_environment(EnvVar.KROKI_PORT),
        "mcp": get_environment(EnvVar.MCP_PORT),
    }


__all__ = [
    # Core types
    "EnvConfig",
    "EnvVar",
    # Main interface
    "get_environment",
    "get_environment_info",
    # Convenience functions
    "get_kroki_url",
    "get_corpus_path",
    "get_data_dir",
    "get_models_dir",
    "get_index_dir",
    "get_available_llm_providers",
    # Introspection
    "list_environment_variables",
    "get_docker_ports",
]
