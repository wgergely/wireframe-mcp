"""Configuration management for corpus directories.

Provides centralized path resolution for all corpus-related storage:
- Data directory: corpus datasets
- Models directory: embedding models
- Index directory: vector indices (future)

All paths follow the same resolution priority:
1. Explicit override parameter
2. Environment variable (CORPUS_{NAME}_DIR)
3. {repo_root}/.corpus/{subdir}
"""

import os
from pathlib import Path


def get_corpus_dir(
    subdir: str,
    env_var: str,
    override: Path | str | None = None,
) -> Path:
    """Generic corpus directory resolver with priority-based resolution.

    Resolution priority:
        1. Explicit override parameter (highest priority)
        2. Environment variable specified by env_var
        3. {repo_root}/.corpus/{subdir} (default)

    Args:
        subdir: Subdirectory name under .corpus (e.g., "data", "models").
        env_var: Environment variable name to check (e.g., "CORPUS_DATA_DIR").
        override: Optional path override. Takes highest priority.

    Returns:
        Resolved Path to the directory.

    Raises:
        RuntimeError: If repository root cannot be detected when using default.

    Examples:
        >>> get_corpus_dir("data", "CORPUS_DATA_DIR")
        PosixPath('/path/to/repo/.corpus/data')

        >>> get_corpus_dir("models", "CORPUS_MODELS_DIR", "/custom/path")
        PosixPath('/custom/path')
    """
    if override is not None:
        return Path(override)

    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value)

    repo_root = _find_repo_root()
    return repo_root / ".corpus" / subdir


def get_data_dir(override: Path | str | None = None) -> Path:
    """Get corpus data directory.

    Resolution priority:
        1. Explicit override parameter
        2. CORPUS_DATA_DIR environment variable
        3. {repo_root}/.corpus/data

    Args:
        override: Optional path override.

    Returns:
        Resolved Path to data directory.
    """
    return get_corpus_dir("data", "CORPUS_DATA_DIR", override)


def _find_repo_root(start_path: Path | None = None) -> Path:
    """Find repository root by searching for .gitignore file.

    Walks up the directory tree from start_path (or cwd) looking for .gitignore.

    Args:
        start_path: Directory to start search from. Defaults to current working
            directory.

    Returns:
        Path to repository root directory containing .gitignore.

    Raises:
        RuntimeError: If .gitignore is not found after reaching filesystem root.

    Examples:
        >>> _find_repo_root()
        PosixPath('/path/to/repo')
    """
    current = start_path if start_path else Path.cwd()
    current = current.resolve()  # Resolve to absolute path

    # Walk up directory tree
    while True:
        gitignore_path = current / ".gitignore"

        if gitignore_path.exists():
            return current

        # Check if we've reached the filesystem root
        parent = current.parent
        if parent == current:
            raise RuntimeError(
                "Could not find repository root. "
                "No .gitignore file found in current directory or any "
                f"parent directories. Started search from: "
                f"{start_path or Path.cwd()}"
            )

        current = parent


def get_models_dir(override: Path | str | None = None) -> Path:
    """Get embedding models directory.

    Resolution priority:
        1. Explicit override parameter
        2. CORPUS_MODELS_DIR environment variable
        3. {repo_root}/.corpus/models

    Args:
        override: Optional path override.

    Returns:
        Resolved Path to models directory.
    """
    return get_corpus_dir("models", "CORPUS_MODELS_DIR", override)


__all__ = ["get_corpus_dir", "get_data_dir", "get_models_dir"]
