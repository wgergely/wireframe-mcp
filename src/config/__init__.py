"""Centralized configuration management for wireframe-mcp.

Provides unified access to all configuration via the `get_environment()` function.

Example:
    >>> from src.config import EnvVar, get_environment
    >>>
    >>> # Get any environment variable with automatic type conversion
    >>> port = get_environment(EnvVar.KROKI_PORT)  # Returns int: 18000
    >>> api_key = get_environment(EnvVar.OPENAI_API_KEY)  # Returns str | None
    >>>
    >>> # Override at runtime
    >>> port = get_environment(EnvVar.KROKI_PORT, override=9000)
    >>>
    >>> # List available variables by category
    >>> docker_vars = list_environment_variables("docker")
    >>> for var in docker_vars:
    ...     info = get_environment_info(var)
    ...     print(f"{info.name}: {info.description}")

Environment Variable Categories:
    llm: API keys for LLM providers (OpenAI, Anthropic, DeepSeek, Qwen)
    embedding: Embedding service configuration (Voyage, local)
    service: Service URLs and hosts (Kroki, Ollama, MCP)
    docker: Docker/container ports (avoids common ports 8000, 8080)
    corpus: Data directory paths
"""

from .lib import (
    # Core types
    EnvConfig,
    EnvVar,
    # Main interface
    get_available_llm_providers,
    get_corpus_path,
    get_data_dir,
    get_default_embedding_backend,
    get_default_llm_model,
    get_docker_ports,
    get_environment,
    get_environment_info,
    get_index_dir,
    get_kroki_url,
    get_models_dir,
    # Introspection
    list_environment_variables,
)

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
    "get_default_llm_model",
    "get_default_embedding_backend",
    "get_docker_ports",
    # Introspection
    "list_environment_variables",
]
