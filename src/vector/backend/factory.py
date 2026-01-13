"""Backend factory for creating embedding backends from model specs.

Provides a unified interface for creating embedding backends
based on model specifications, enabling easy provider switching.
"""

from .base import EmbeddingBackend
from .model_spec import (
    EmbeddingModel,
    ModelSpec,
    ProviderType,
    get_model_spec,
)


def create_backend(
    model: str | EmbeddingModel | ModelSpec = EmbeddingModel.LOCAL_MINILM,
    *,
    api_key: str | None = None,
    device: str | None = None,
    **kwargs,
) -> EmbeddingBackend:
    """Create an embedding backend from a model specification.

    This is the primary factory function for creating embedding backends.
    It automatically selects the correct backend class based on the
    model's provider type.

    Args:
        model: Model to use. Can be:
            - String model name (e.g., "voyage-3", "all-MiniLM-L6-v2")
            - EmbeddingModel enum value
            - ModelSpec instance
        api_key: API key for remote providers (Voyage).
        device: Device for local models ('cuda', 'cpu', or None for auto).
        **kwargs: Additional arguments passed to backend constructor.

    Returns:
        Configured EmbeddingBackend instance.

    Raises:
        ValueError: If model is unknown or configuration is invalid.

    Example:
        >>> # Use Voyage API
        >>> backend = create_backend(EmbeddingModel.VOYAGE_3, api_key="...")
        >>> vectors = backend.embed(["hello world"])

        >>> # Use local model with GPU
        >>> backend = create_backend("all-MiniLM-L6-v2", device="cuda")
        >>> vectors = backend.embed(["hello world"])

        >>> # Switch providers easily
        >>> for model in [EmbeddingModel.VOYAGE_3, EmbeddingModel.LOCAL_MINILM]:
        ...     backend = create_backend(model)
        ...     print(f"{model.spec.name}: {backend.dimension}D")
    """
    spec = get_model_spec(model)

    if spec.provider == ProviderType.VOYAGE:
        from .voyage import VoyageBackend

        return VoyageBackend(
            api_key=api_key,
            model=spec.name,
            **kwargs,
        )
    elif spec.provider == ProviderType.LOCAL:
        from .local import LocalBackend

        return LocalBackend(
            model_name=spec.name,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider type: {spec.provider}")


__all__ = ["create_backend"]
