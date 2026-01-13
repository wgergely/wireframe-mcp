"""Tests for the centralized model manager."""

import pytest

from .lib import (
    DEFAULT_LOCAL_MODEL,
    EmbeddingModel,
    ModelManager,
    get_model_manager,
)


@pytest.mark.unit
def test_model_manager_uses_custom_models_dir(tmp_path):
    """ModelManager uses provided models directory."""
    custom_dir = tmp_path / "custom_models"
    manager = ModelManager(models_dir=custom_dir)

    assert manager.models_dir == custom_dir


@pytest.mark.unit
def test_model_manager_ensure_dir_creates_directory(tmp_path):
    """ensure_dir creates the models directory."""
    models_dir = tmp_path / "new_models"
    manager = ModelManager(models_dir=models_dir)

    assert not models_dir.exists()
    manager.ensure_dir()
    assert models_dir.exists()


@pytest.mark.unit
def test_get_model_path_returns_correct_path(tmp_path):
    """get_model_path returns path within models directory."""
    manager = ModelManager(models_dir=tmp_path)

    result = manager.get_model_path("all-MiniLM-L6-v2")

    assert result == tmp_path / "all-MiniLM-L6-v2"


@pytest.mark.unit
def test_is_downloaded_returns_false_when_not_present(tmp_path):
    """is_downloaded returns False when model not downloaded."""
    manager = ModelManager(models_dir=tmp_path)

    assert manager.is_downloaded("all-MiniLM-L6-v2") is False


@pytest.mark.unit
def test_is_downloaded_returns_true_when_present(tmp_path):
    """is_downloaded returns True when model config exists."""
    manager = ModelManager(models_dir=tmp_path)
    model_dir = tmp_path / "all-MiniLM-L6-v2"
    model_dir.mkdir()
    (model_dir / "config.json").touch()

    assert manager.is_downloaded("all-MiniLM-L6-v2") is True


@pytest.mark.unit
def test_list_downloaded_returns_empty_when_no_models(tmp_path):
    """list_downloaded returns empty list when no models."""
    manager = ModelManager(models_dir=tmp_path)

    assert manager.list_downloaded() == []


@pytest.mark.unit
def test_list_downloaded_returns_downloaded_models(tmp_path):
    """list_downloaded returns list of downloaded models."""
    manager = ModelManager(models_dir=tmp_path)

    # Create fake downloaded models
    for name in ["model-a", "model-b"]:
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / "config.json").touch()

    # Create a directory without config.json (not a valid model)
    (tmp_path / "not-a-model").mkdir()

    downloaded = manager.list_downloaded()

    assert set(downloaded) == {"model-a", "model-b"}


@pytest.mark.unit
def test_get_dimension_returns_known_dimensions():
    """get_dimension returns dimension from registry."""
    manager = ModelManager()

    for model in EmbeddingModel.list_local():
        spec = model.spec
        assert manager.get_dimension(spec.name) == spec.dimension


@pytest.mark.unit
def test_get_dimension_returns_none_for_unknown_model():
    """get_dimension returns None for unknown model."""
    manager = ModelManager()

    assert manager.get_dimension("unknown-model") is None


@pytest.mark.unit
def test_delete_removes_model_directory(tmp_path):
    """delete removes the model directory."""
    manager = ModelManager(models_dir=tmp_path)
    model_dir = tmp_path / "model-to-delete"
    model_dir.mkdir()
    (model_dir / "config.json").touch()

    result = manager.delete("model-to-delete")

    assert result is True
    assert not model_dir.exists()


@pytest.mark.unit
def test_delete_returns_false_when_not_found(tmp_path):
    """delete returns False when model not found."""
    manager = ModelManager(models_dir=tmp_path)

    result = manager.delete("nonexistent-model")

    assert result is False


@pytest.mark.unit
def test_default_model_is_in_registry():
    """DEFAULT_LOCAL_MODEL is a valid model in the registry."""
    manager = ModelManager()
    spec = manager.get_spec(DEFAULT_LOCAL_MODEL.spec.name)
    assert spec is not None


@pytest.mark.unit
def test_model_spec_has_required_fields():
    """All local models have required ModelSpec fields."""
    for model in EmbeddingModel.list_local():
        spec = model.spec
        assert spec.name, f"{model} missing name"
        assert spec.dimension > 0, f"{model} missing dimension"
        assert spec.description, f"{model} missing description"
        assert spec.size_mb is not None, f"{model} missing size_mb"


@pytest.mark.unit
def test_get_model_manager_returns_singleton():
    """get_model_manager returns a singleton instance."""
    import src.vector.models.lib as lib_module

    # Reset singleton for test
    original = lib_module._default_manager
    lib_module._default_manager = None

    try:
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        assert manager1 is manager2
    finally:
        lib_module._default_manager = original


@pytest.mark.integration
def test_download_creates_model_directory(tmp_path):
    """download creates model files in the models directory."""
    manager = ModelManager(models_dir=tmp_path)

    # This test requires sentence-transformers and network access
    pytest.importorskip("sentence_transformers")

    model_path = manager.download(DEFAULT_LOCAL_MODEL.spec.name)

    assert model_path.exists()
    assert (model_path / "config.json").exists()


@pytest.mark.integration
def test_load_returns_model_instance(tmp_path):
    """load returns a SentenceTransformer model."""
    manager = ModelManager(models_dir=tmp_path)

    # This test requires sentence-transformers and network access
    st = pytest.importorskip("sentence_transformers")

    model = manager.load(DEFAULT_LOCAL_MODEL.spec.name, auto_download=True)

    assert isinstance(model, st.SentenceTransformer)


@pytest.mark.integration
def test_load_caches_model_instance(tmp_path):
    """load returns cached model on subsequent calls."""
    manager = ModelManager(models_dir=tmp_path)

    pytest.importorskip("sentence_transformers")

    model1 = manager.load(DEFAULT_LOCAL_MODEL.spec.name, auto_download=True)
    model2 = manager.load(DEFAULT_LOCAL_MODEL.spec.name, auto_download=True)

    assert model1 is model2


@pytest.mark.unit
def test_load_raises_file_not_found_when_no_auto_download(tmp_path):
    """load raises FileNotFoundError when model missing and auto_download=False."""
    manager = ModelManager(models_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        manager.load("all-MiniLM-L6-v2", auto_download=False)
