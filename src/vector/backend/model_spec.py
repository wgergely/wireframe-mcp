"""Model specification system for embedding backends.

Provides a unified way to define and select embedding models across
different providers (Voyage, Local sentence-transformers).
"""

from dataclasses import dataclass, field
from enum import Enum


class ModelCapability(Enum):
    """Capabilities that an embedding model may support."""

    BATCH_EMBED = "batch_embed"
    QUERY_EMBED = "query_embed"
    GPU_ACCELERATED = "gpu_accelerated"
    OFFLINE = "offline"
    LARGE_CONTEXT = "large_context"


class ProviderType(Enum):
    """Available embedding backend providers."""

    VOYAGE = "voyage"
    LOCAL = "local"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an embedding model.

    Provides metadata about a model's capabilities, dimensions, and
    provider requirements. Used to configure embedding backends.

    Attributes:
        name: Model identifier (e.g., "voyage-3", "all-MiniLM-L6-v2").
        dimension: Output embedding vector dimension.
        provider: Which backend provider to use.
        max_tokens: Maximum input context length.
        capabilities: Set of supported capabilities.
        description: Human-readable description.
        requires_api_key: Whether this model needs an API key.
        size_mb: Approximate model size in megabytes (for local models).
    """

    name: str
    dimension: int
    provider: ProviderType
    max_tokens: int
    capabilities: frozenset[ModelCapability] = field(default_factory=frozenset)
    description: str = ""
    requires_api_key: bool = False
    size_mb: int | None = None

    def supports(self, capability: ModelCapability) -> bool:
        """Check if model supports a capability."""
        return capability in self.capabilities

    @property
    def is_local(self) -> bool:
        """Check if model runs locally."""
        return self.provider == ProviderType.LOCAL

    @property
    def is_remote(self) -> bool:
        """Check if model uses remote API."""
        return self.provider == ProviderType.VOYAGE


class EmbeddingModel(Enum):
    """Registry of available embedding models.

    Provides predefined model specifications for easy selection.
    Use .spec property to get the full ModelSpec.

    Example:
        >>> model = EmbeddingModel.VOYAGE_3
        >>> print(model.spec.dimension)
        1024
        >>> backend = create_backend(model)
    """

    # Voyage AI models (remote, high quality)
    VOYAGE_3 = ModelSpec(
        name="voyage-3",
        dimension=1024,
        provider=ProviderType.VOYAGE,
        max_tokens=32000,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.LARGE_CONTEXT,
            }
        ),
        description="Voyage AI flagship model, best quality",
        requires_api_key=True,
    )

    VOYAGE_3_LITE = ModelSpec(
        name="voyage-3-lite",
        dimension=512,
        provider=ProviderType.VOYAGE,
        max_tokens=32000,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.LARGE_CONTEXT,
            }
        ),
        description="Voyage AI lite model, faster/cheaper",
        requires_api_key=True,
    )

    VOYAGE_CODE_3 = ModelSpec(
        name="voyage-code-3",
        dimension=1024,
        provider=ProviderType.VOYAGE,
        max_tokens=32000,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.LARGE_CONTEXT,
            }
        ),
        description="Voyage AI code-optimized model",
        requires_api_key=True,
    )

    # Local sentence-transformers models (offline, GPU optional)
    LOCAL_MINILM = ModelSpec(
        name="all-MiniLM-L6-v2",
        dimension=384,
        provider=ProviderType.LOCAL,
        max_tokens=512,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.GPU_ACCELERATED,
                ModelCapability.OFFLINE,
            }
        ),
        description="Fast general-purpose model, good balance",
        requires_api_key=False,
        size_mb=80,
    )

    LOCAL_MPNET = ModelSpec(
        name="all-mpnet-base-v2",
        dimension=768,
        provider=ProviderType.LOCAL,
        max_tokens=512,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.GPU_ACCELERATED,
                ModelCapability.OFFLINE,
            }
        ),
        description="Higher quality general-purpose model",
        requires_api_key=False,
        size_mb=420,
    )

    LOCAL_QA = ModelSpec(
        name="multi-qa-MiniLM-L6-cos-v1",
        dimension=384,
        provider=ProviderType.LOCAL,
        max_tokens=512,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.GPU_ACCELERATED,
                ModelCapability.OFFLINE,
            }
        ),
        description="Optimized for semantic search/QA",
        requires_api_key=False,
        size_mb=80,
    )

    LOCAL_PARAPHRASE = ModelSpec(
        name="paraphrase-MiniLM-L6-v2",
        dimension=384,
        provider=ProviderType.LOCAL,
        max_tokens=512,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.GPU_ACCELERATED,
                ModelCapability.OFFLINE,
            }
        ),
        description="Optimized for paraphrase detection",
        requires_api_key=False,
        size_mb=80,
    )

    LOCAL_DISTILROBERTA = ModelSpec(
        name="all-distilroberta-v1",
        dimension=768,
        provider=ProviderType.LOCAL,
        max_tokens=512,
        capabilities=frozenset(
            {
                ModelCapability.BATCH_EMBED,
                ModelCapability.QUERY_EMBED,
                ModelCapability.GPU_ACCELERATED,
                ModelCapability.OFFLINE,
            }
        ),
        description="DistilRoBERTa-based general embeddings",
        requires_api_key=False,
        size_mb=290,
    )

    @property
    def spec(self) -> ModelSpec:
        """Get the ModelSpec for this model."""
        return self.value

    @classmethod
    def by_name(cls, name: str) -> "EmbeddingModel | None":
        """Look up model by name string.

        Args:
            name: Model name to find.

        Returns:
            EmbeddingModel if found, None otherwise.
        """
        for model in cls:
            if model.spec.name == name:
                return model
        return None

    @classmethod
    def list_local(cls) -> list["EmbeddingModel"]:
        """Get all local (offline) models."""
        return [m for m in cls if m.spec.is_local]

    @classmethod
    def list_remote(cls) -> list["EmbeddingModel"]:
        """Get all remote (API) models."""
        return [m for m in cls if m.spec.is_remote]


# Default models for each provider
DEFAULT_LOCAL_MODEL = EmbeddingModel.LOCAL_MINILM
DEFAULT_VOYAGE_MODEL = EmbeddingModel.VOYAGE_3


def get_model_spec(
    model: str | EmbeddingModel | ModelSpec,
) -> ModelSpec:
    """Resolve a model reference to its ModelSpec.

    Args:
        model: Model name string, EmbeddingModel enum, or ModelSpec.

    Returns:
        Resolved ModelSpec.

    Raises:
        ValueError: If model name not found in registry.
    """
    if isinstance(model, ModelSpec):
        return model
    if isinstance(model, EmbeddingModel):
        return model.spec
    # String lookup
    found = EmbeddingModel.by_name(model)
    if found:
        return found.spec
    raise ValueError(f"Unknown model: {model}")


__all__ = [
    "ModelCapability",
    "ProviderType",
    "ModelSpec",
    "EmbeddingModel",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_VOYAGE_MODEL",
    "get_model_spec",
]
