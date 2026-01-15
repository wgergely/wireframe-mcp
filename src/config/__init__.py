"""Configuration management for corpus directories.

Provides centralized path resolution for all corpus-related storage.

Environment Variables:
    CORPUS_DATA_DIR: Override corpus data directory ({repo_root}/.corpus/data).
    CORPUS_MODELS_DIR: Override models directory ({repo_root}/.corpus/models).
    CORPUS_INDEX_DIR: Override index directory ({repo_root}/.corpus/index).

Examples:
    >>> from src.config import get_data_dir, get_models_dir, get_index_dir
    >>>
    >>> # Convenience functions
    >>> data_dir = get_data_dir()
    >>> models_dir = get_models_dir()
    >>> index_dir = get_index_dir()
    >>>
    >>> # Generic API for custom subdirs
    >>> cache_dir = get_corpus_dir("cache", "CORPUS_CACHE_DIR")
"""

from .lib import get_corpus_dir, get_data_dir, get_index_dir, get_models_dir

__all__ = ["get_corpus_dir", "get_data_dir", "get_index_dir", "get_models_dir"]
