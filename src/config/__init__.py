"""Configuration management for corpus directories.

Provides centralized path resolution for all corpus-related storage.

Environment Variables:
    CORPUS_DATA_DIR: Override corpus data directory ({repo_root}/.corpus/data).
    CORPUS_MODELS_DIR: Override models directory ({repo_root}/.corpus/models).

Examples:
    >>> from src.config import get_data_dir, get_models_dir, get_corpus_dir
    >>>
    >>> # Convenience functions
    >>> data_dir = get_data_dir()
    >>> models_dir = get_models_dir()
    >>>
    >>> # Generic API for custom subdirs
    >>> index_dir = get_corpus_dir("index", "CORPUS_INDEX_DIR")
"""

from .lib import get_corpus_dir, get_data_dir, get_models_dir

__all__ = ["get_corpus_dir", "get_data_dir", "get_models_dir"]
