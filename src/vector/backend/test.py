"""Tests for model specification system."""

import pytest

from src.vector.backend.model_spec import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_VOYAGE_MODEL,
    EmbeddingModel,
    ModelCapability,
    ModelSpec,
    ProviderType,
    get_model_spec,
)


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    @pytest.mark.unit
    def test_model_spec_creation(self):
        """Test creating a ModelSpec."""
        spec = ModelSpec(
            name="test-model",
            dimension=512,
            provider=ProviderType.LOCAL,
            max_tokens=256,
        )
        assert spec.name == "test-model"
        assert spec.dimension == 512
        assert spec.provider == ProviderType.LOCAL
        assert spec.max_tokens == 256

    @pytest.mark.unit
    def test_model_spec_capabilities(self):
        """Test capability checking."""
        spec = ModelSpec(
            name="test",
            dimension=384,
            provider=ProviderType.LOCAL,
            max_tokens=512,
            capabilities=frozenset(
                {ModelCapability.OFFLINE, ModelCapability.BATCH_EMBED}
            ),
        )
        assert spec.supports(ModelCapability.OFFLINE)
        assert spec.supports(ModelCapability.BATCH_EMBED)
        assert not spec.supports(ModelCapability.LARGE_CONTEXT)

    @pytest.mark.unit
    def test_is_local_and_remote(self):
        """Test local/remote detection."""
        local_spec = ModelSpec(
            name="local",
            dimension=384,
            provider=ProviderType.LOCAL,
            max_tokens=512,
        )
        assert local_spec.is_local
        assert not local_spec.is_remote

        remote_spec = ModelSpec(
            name="remote",
            dimension=1024,
            provider=ProviderType.VOYAGE,
            max_tokens=32000,
        )
        assert not remote_spec.is_local
        assert remote_spec.is_remote


class TestEmbeddingModel:
    """Tests for EmbeddingModel enum registry."""

    @pytest.mark.unit
    def test_voyage_models_exist(self):
        """Test that Voyage models are defined."""
        assert EmbeddingModel.VOYAGE_3.spec.name == "voyage-3"
        assert EmbeddingModel.VOYAGE_3.spec.dimension == 1024
        assert EmbeddingModel.VOYAGE_3.spec.requires_api_key

    @pytest.mark.unit
    def test_local_models_exist(self):
        """Test that local models are defined."""
        assert EmbeddingModel.LOCAL_MINILM.spec.name == "all-MiniLM-L6-v2"
        assert EmbeddingModel.LOCAL_MINILM.spec.dimension == 384
        assert not EmbeddingModel.LOCAL_MINILM.spec.requires_api_key

    @pytest.mark.unit
    def test_by_name_lookup(self):
        """Test looking up models by name."""
        model = EmbeddingModel.by_name("voyage-3")
        assert model == EmbeddingModel.VOYAGE_3

        model = EmbeddingModel.by_name("all-MiniLM-L6-v2")
        assert model == EmbeddingModel.LOCAL_MINILM

        model = EmbeddingModel.by_name("nonexistent")
        assert model is None

    @pytest.mark.unit
    def test_list_local(self):
        """Test listing local models."""
        local_models = EmbeddingModel.list_local()
        assert len(local_models) >= 3
        assert all(m.spec.is_local for m in local_models)

    @pytest.mark.unit
    def test_list_remote(self):
        """Test listing remote models."""
        remote_models = EmbeddingModel.list_remote()
        assert len(remote_models) >= 3
        assert all(m.spec.is_remote for m in remote_models)


class TestGetModelSpec:
    """Tests for get_model_spec helper."""

    @pytest.mark.unit
    def test_from_string(self):
        """Test resolving from string name."""
        spec = get_model_spec("voyage-3")
        assert spec.name == "voyage-3"
        assert spec.dimension == 1024

    @pytest.mark.unit
    def test_from_enum(self):
        """Test resolving from enum."""
        spec = get_model_spec(EmbeddingModel.LOCAL_MINILM)
        assert spec.name == "all-MiniLM-L6-v2"

    @pytest.mark.unit
    def test_from_spec(self):
        """Test passthrough of ModelSpec."""
        original = ModelSpec(
            name="custom",
            dimension=256,
            provider=ProviderType.LOCAL,
            max_tokens=128,
        )
        resolved = get_model_spec(original)
        assert resolved is original

    @pytest.mark.unit
    def test_unknown_model_raises(self):
        """Test that unknown model names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_spec("nonexistent-model")


class TestDefaults:
    """Tests for default model constants."""

    @pytest.mark.unit
    def test_default_local_model(self):
        """Test default local model is set."""
        assert DEFAULT_LOCAL_MODEL == EmbeddingModel.LOCAL_MINILM

    @pytest.mark.unit
    def test_default_voyage_model(self):
        """Test default Voyage model is set."""
        assert DEFAULT_VOYAGE_MODEL == EmbeddingModel.VOYAGE_3


class TestCreateBackend:
    """Tests for create_backend factory function."""

    @pytest.mark.unit
    def test_create_local_backend_from_enum(self):
        """Test creating local backend from enum."""
        from src.vector.backend import create_backend
        from src.vector.backend.local import LocalBackend

        backend = create_backend(EmbeddingModel.LOCAL_MINILM)
        assert isinstance(backend, LocalBackend)
        assert backend.name == "local:all-MiniLM-L6-v2"

    @pytest.mark.unit
    def test_create_local_backend_from_string(self):
        """Test creating local backend from string name."""
        from src.vector.backend import create_backend
        from src.vector.backend.local import LocalBackend

        backend = create_backend("all-MiniLM-L6-v2")
        assert isinstance(backend, LocalBackend)

    @pytest.mark.unit
    def test_create_voyage_backend_requires_api_key(self):
        """Test that Voyage backend requires API key."""
        import os

        from src.vector.backend import create_backend

        # Remove any existing API key from environment
        original = os.environ.pop("VOYAGE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                create_backend(EmbeddingModel.VOYAGE_3)
        finally:
            if original:
                os.environ["VOYAGE_API_KEY"] = original

    @pytest.mark.unit
    def test_create_voyage_backend_with_api_key(self):
        """Test creating Voyage backend with API key."""
        from src.vector.backend import create_backend
        from src.vector.backend.voyage import VoyageBackend

        backend = create_backend(EmbeddingModel.VOYAGE_3, api_key="test-key")
        assert isinstance(backend, VoyageBackend)
        assert backend.dimension == 1024

    @pytest.mark.unit
    def test_factory_respects_device_kwarg(self):
        """Test that device kwarg is passed to local backend."""
        from src.vector.backend import create_backend

        # Just verify no error - actual device selection depends on hardware
        backend = create_backend(EmbeddingModel.LOCAL_MINILM, device="cpu")
        assert backend is not None
