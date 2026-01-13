"""Centralized embedding model manager.

Provides unified storage and retrieval for local embedding models.
Models are stored in {repo_root}/.corpus/models by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import get_models_dir

from ..backend.model_spec import DEFAULT_LOCAL_MODEL, EmbeddingModel, ModelSpec

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _get_local_model_registry() -> dict[str, ModelSpec]:
    """Build registry of local models from EmbeddingModel enum."""
    return {m.spec.name: m.spec for m in EmbeddingModel.list_local()}


class ModelManager:
    """Centralized manager for local embedding models.

    Handles model downloading, caching, and loading with a unified
    storage location at {repo_root}/.corpus/models.

    Example:
        >>> manager = ModelManager()
        >>> model = manager.load("all-MiniLM-L6-v2")
        >>> embeddings = model.encode(["hello world"])
    """

    def __init__(self, models_dir: Path | str | None = None):
        """Initialize model manager.

        Args:
            models_dir: Override path for model storage. Uses get_models_dir()
                if not provided.
        """
        self._models_dir = get_models_dir(models_dir)
        self._loaded_models: dict[str, SentenceTransformer] = {}
        self._registry = _get_local_model_registry()

    @property
    def models_dir(self) -> Path:
        """Get the models storage directory."""
        return self._models_dir

    def ensure_dir(self) -> None:
        """Ensure models directory exists."""
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get local storage path for a model.

        Args:
            model_name: Name of the model (e.g., "all-MiniLM-L6-v2").

        Returns:
            Path where model is/will be stored.
        """
        return self._models_dir / model_name

    def is_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded.

        Args:
            model_name: Name of the model.

        Returns:
            True if model exists locally.
        """
        model_path = self.get_model_path(model_name)
        # Check for model artifacts (config.json is always present)
        return (model_path / "config.json").exists()

    def list_downloaded(self) -> list[str]:
        """List all downloaded models.

        Returns:
            List of model names that are downloaded.
        """
        if not self._models_dir.exists():
            return []

        downloaded = []
        for path in self._models_dir.iterdir():
            if path.is_dir() and (path / "config.json").exists():
                downloaded.append(path.name)
        return downloaded

    def get_spec(self, model_name: str) -> ModelSpec | None:
        """Get ModelSpec for a model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelSpec if found, None otherwise.
        """
        return self._registry.get(model_name)

    def get_dimension(self, model_name: str) -> int | None:
        """Get embedding dimension for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Embedding dimension, or None if unknown.
        """
        spec = self.get_spec(model_name)
        return spec.dimension if spec else None

    def get_size_mb(self, model_name: str) -> int | None:
        """Get approximate model size in MB.

        Args:
            model_name: Name of the model.

        Returns:
            Size in MB if known, None otherwise.
        """
        spec = self.get_spec(model_name)
        return spec.size_mb if spec else None

    def download(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL.spec.name,
        force: bool = False,
    ) -> Path:
        """Download a model to the centralized storage.

        Args:
            model_name: sentence-transformers model name.
            force: Re-download even if already exists.

        Returns:
            Path to downloaded model.

        Raises:
            ImportError: If sentence-transformers not installed.
        """
        model_path = self.get_model_path(model_name)

        if not force and self.is_downloaded(model_name):
            logger.info(f"Model '{model_name}' already exists at {model_path}")
            return model_path

        self.ensure_dir()
        logger.info(f"Downloading model '{model_name}' to {model_path}")

        try:
            from sentence_transformers import SentenceTransformer

            # Download model with explicit cache folder
            # This downloads to our centralized location
            model = SentenceTransformer(
                model_name,
                cache_folder=str(self._models_dir),
            )
            # Save to ensure all files are in place
            model.save(str(model_path))
            logger.info(f"Model '{model_name}' downloaded successfully")
            return model_path

        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for model download. "
                "Install with: pip install sentence-transformers"
            ) from e

    def load(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL.spec.name,
        device: str | None = None,
        auto_download: bool = True,
    ) -> SentenceTransformer:
        """Load a model for inference.

        Args:
            model_name: Model name to load.
            device: Device for inference ('cuda', 'cpu', or None for auto).
            auto_download: Download model if not present.

        Returns:
            Loaded SentenceTransformer model.

        Raises:
            ImportError: If sentence-transformers not installed.
            FileNotFoundError: If model not found and auto_download=False.
        """
        # Return cached model if already loaded
        cache_key = f"{model_name}:{device}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        model_path = self.get_model_path(model_name)

        # Check if model exists
        if not self.is_downloaded(model_name):
            if auto_download:
                self.download(model_name)
            else:
                raise FileNotFoundError(
                    f"Model '{model_name}' not found at {model_path}. "
                    f"Run `manager.download('{model_name}')` first."
                )

        try:
            from sentence_transformers import SentenceTransformer

            # Load from local path
            model = SentenceTransformer(str(model_path), device=device)
            self._loaded_models[cache_key] = model
            logger.info(f"Loaded model '{model_name}' on device '{model.device}'")
            return model

        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for model loading. "
                "Install with: pip install sentence-transformers"
            ) from e

    def delete(self, model_name: str) -> bool:
        """Delete a downloaded model.

        Args:
            model_name: Name of model to delete.

        Returns:
            True if deleted, False if not found.
        """
        import shutil

        model_path = self.get_model_path(model_name)
        if model_path.exists():
            shutil.rmtree(model_path)
            # Clear from cache
            keys_to_remove = [
                k for k in self._loaded_models if k.startswith(model_name)
            ]
            for key in keys_to_remove:
                del self._loaded_models[key]
            logger.info(f"Deleted model '{model_name}' from {model_path}")
            return True
        return False


# Module-level singleton for convenience
_default_manager: ModelManager | None = None


def get_model_manager(models_dir: Path | str | None = None) -> ModelManager:
    """Get or create the default model manager.

    Args:
        models_dir: Override models directory. Only used on first call.

    Returns:
        ModelManager instance.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager(models_dir)
    return _default_manager


__all__ = [
    "ModelManager",
    "get_model_manager",
    # Re-export from model_spec for convenience
    "DEFAULT_LOCAL_MODEL",
    "EmbeddingModel",
    "ModelSpec",
]
