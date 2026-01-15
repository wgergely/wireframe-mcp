"""Tests for configuration management."""

import re

import pytest

from .lib import _find_repo_root, get_corpus_dir, get_data_dir, get_models_dir


@pytest.mark.unit
def test_get_data_dir_with_override(tmp_path):
    """Override parameter takes highest priority."""
    custom_path = tmp_path / "custom_override"
    result = get_data_dir(custom_path)

    assert result == custom_path


@pytest.mark.unit
def test_get_data_dir_with_override_as_string(tmp_path):
    """Override parameter accepts string paths."""
    custom_path = tmp_path / "custom_string"
    result = get_data_dir(str(custom_path))

    assert result == custom_path


@pytest.mark.unit
def test_get_data_dir_with_env_var(tmp_path, monkeypatch):
    """CORPUS_DATA_DIR env var used when no override provided."""
    env_path = tmp_path / "from_env"
    monkeypatch.setenv("CORPUS_DATA_DIR", str(env_path))

    result = get_data_dir()

    assert result == env_path


@pytest.mark.unit
def test_get_data_dir_override_beats_env_var(tmp_path, monkeypatch):
    """Explicit override takes precedence over environment variable."""
    env_path = tmp_path / "from_env"
    override_path = tmp_path / "override"

    monkeypatch.setenv("CORPUS_DATA_DIR", str(env_path))
    result = get_data_dir(override_path)

    assert result == override_path


@pytest.mark.unit
def test_get_data_dir_default_finds_repo_root(tmp_path, monkeypatch):
    """Default behavior finds repo root and returns .corpus/data."""
    # Create fake repo structure
    repo_root = tmp_path / "fake_repo"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    # Create subdirectory to run from
    subdir = repo_root / "src" / "submodule"
    subdir.mkdir(parents=True)

    # Change to subdirectory
    monkeypatch.chdir(subdir)

    # Clear env var to force default behavior
    monkeypatch.delenv("CORPUS_DATA_DIR", raising=False)

    result = get_data_dir()

    expected = repo_root / ".corpus" / "data"
    assert result == expected


@pytest.mark.unit
def test_find_repo_root_from_cwd(tmp_path, monkeypatch):
    """Repository root detection finds .gitignore from current directory."""
    repo_root = tmp_path / "test_repo"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    monkeypatch.chdir(repo_root)

    result = _find_repo_root()

    assert result == repo_root


@pytest.mark.unit
def test_find_repo_root_from_subdirectory(tmp_path, monkeypatch):
    """Repository root detection walks up from subdirectory."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    deep_subdir = repo_root / "a" / "b" / "c"
    deep_subdir.mkdir(parents=True)

    monkeypatch.chdir(deep_subdir)

    result = _find_repo_root()

    assert result == repo_root


@pytest.mark.unit
def test_find_repo_root_with_explicit_start_path(tmp_path):
    """Repository root detection accepts explicit start path."""
    repo_root = tmp_path / "explicit_repo"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    subdir = repo_root / "src"
    subdir.mkdir()

    result = _find_repo_root(start_path=subdir)

    assert result == repo_root


@pytest.mark.unit
def test_find_repo_root_not_found_raises_error(tmp_path, monkeypatch):
    """RuntimeError raised when .gitignore not found."""
    # Create directory without .gitignore
    no_repo = tmp_path / "not_a_repo"
    no_repo.mkdir()

    monkeypatch.chdir(no_repo)

    with pytest.raises(RuntimeError, match="Could not find repository root"):
        _find_repo_root()


@pytest.mark.unit
def test_find_repo_root_error_message_includes_start_path(tmp_path):
    """Error message includes the start path for debugging."""
    no_repo = tmp_path / "no_git_here"
    no_repo.mkdir()

    with pytest.raises(RuntimeError, match=re.escape(str(no_repo))):
        _find_repo_root(start_path=no_repo)


@pytest.mark.unit
def test_get_data_dir_with_none_override_uses_default(tmp_path, monkeypatch):
    """Passing None as override uses default resolution logic."""
    repo_root = tmp_path / "repo_none_test"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("CORPUS_DATA_DIR", raising=False)

    result = get_data_dir(override=None)

    expected = repo_root / ".corpus" / "data"
    assert result == expected


# Tests for get_models_dir


@pytest.mark.unit
def test_get_models_dir_with_override(tmp_path):
    """Override parameter takes highest priority."""
    custom_path = tmp_path / "custom_models"
    result = get_models_dir(custom_path)

    assert result == custom_path


@pytest.mark.unit
def test_get_models_dir_with_override_as_string(tmp_path):
    """Override parameter accepts string paths."""
    custom_path = tmp_path / "custom_models_str"
    result = get_models_dir(str(custom_path))

    assert result == custom_path


@pytest.mark.unit
def test_get_models_dir_with_env_var(tmp_path, monkeypatch):
    """CORPUS_MODELS_DIR env var used when no override provided."""
    env_path = tmp_path / "models_from_env"
    monkeypatch.setenv("CORPUS_MODELS_DIR", str(env_path))

    result = get_models_dir()

    assert result == env_path


@pytest.mark.unit
def test_get_models_dir_override_beats_env_var(tmp_path, monkeypatch):
    """Explicit override takes precedence over environment variable."""
    env_path = tmp_path / "models_env"
    override_path = tmp_path / "models_override"

    monkeypatch.setenv("CORPUS_MODELS_DIR", str(env_path))
    result = get_models_dir(override_path)

    assert result == override_path


@pytest.mark.unit
def test_get_models_dir_default_finds_repo_root(tmp_path, monkeypatch):
    """Default behavior finds repo root and returns .corpus/models."""
    repo_root = tmp_path / "fake_repo_models"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    subdir = repo_root / "src" / "submodule"
    subdir.mkdir(parents=True)

    monkeypatch.chdir(subdir)
    monkeypatch.delenv("CORPUS_MODELS_DIR", raising=False)

    result = get_models_dir()

    expected = repo_root / ".corpus" / "models"
    assert result == expected


# Tests for get_corpus_dir (generic API)


@pytest.mark.unit
def test_get_corpus_dir_with_override(tmp_path):
    """Override parameter takes highest priority."""
    custom_path = tmp_path / "custom_subdir"
    result = get_corpus_dir("index", "CORPUS_INDEX_DIR", custom_path)

    assert result == custom_path


@pytest.mark.unit
def test_get_corpus_dir_with_env_var(tmp_path, monkeypatch):
    """Environment variable used when no override provided."""
    env_path = tmp_path / "env_subdir"
    monkeypatch.setenv("CORPUS_CUSTOM_DIR", str(env_path))

    result = get_corpus_dir("custom", "CORPUS_CUSTOM_DIR")

    assert result == env_path


@pytest.mark.unit
def test_get_corpus_dir_override_beats_env_var(tmp_path, monkeypatch):
    """Override takes precedence over environment variable."""
    env_path = tmp_path / "env_path"
    override_path = tmp_path / "override_path"

    monkeypatch.setenv("CORPUS_TEST_DIR", str(env_path))
    result = get_corpus_dir("test", "CORPUS_TEST_DIR", override_path)

    assert result == override_path


@pytest.mark.unit
def test_get_corpus_dir_default_uses_subdir(tmp_path, monkeypatch):
    """Default behavior uses subdir under .corpus."""
    repo_root = tmp_path / "repo_corpus"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("CORPUS_INDEX_DIR", raising=False)

    result = get_corpus_dir("index", "CORPUS_INDEX_DIR")

    expected = repo_root / ".corpus" / "index"
    assert result == expected


@pytest.mark.unit
def test_get_corpus_dir_custom_subdir(tmp_path, monkeypatch):
    """Custom subdirectory names work correctly."""
    repo_root = tmp_path / "repo_custom"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("CORPUS_CACHE_DIR", raising=False)

    result = get_corpus_dir("cache/embeddings", "CORPUS_CACHE_DIR")

    expected = repo_root / ".corpus" / "cache/embeddings"
    assert result == expected


# Tests for get_index_dir


@pytest.mark.unit
def test_get_index_dir_with_override(tmp_path):
    """Override parameter takes highest priority."""
    from .lib import get_index_dir

    custom_path = tmp_path / "custom_index"
    result = get_index_dir(custom_path)

    assert result == custom_path


@pytest.mark.unit
def test_get_index_dir_with_override_as_string(tmp_path):
    """Override parameter accepts string paths."""
    from .lib import get_index_dir

    custom_path = tmp_path / "custom_index_str"
    result = get_index_dir(str(custom_path))

    assert result == custom_path


@pytest.mark.unit
def test_get_index_dir_with_env_var(tmp_path, monkeypatch):
    """CORPUS_INDEX_DIR env var used when no override provided."""
    from .lib import get_index_dir

    env_path = tmp_path / "index_from_env"
    monkeypatch.setenv("CORPUS_INDEX_DIR", str(env_path))

    result = get_index_dir()

    assert result == env_path


@pytest.mark.unit
def test_get_index_dir_override_beats_env_var(tmp_path, monkeypatch):
    """Explicit override takes precedence over environment variable."""
    from .lib import get_index_dir

    env_path = tmp_path / "index_env"
    override_path = tmp_path / "index_override"

    monkeypatch.setenv("CORPUS_INDEX_DIR", str(env_path))
    result = get_index_dir(override_path)

    assert result == override_path


@pytest.mark.unit
def test_get_index_dir_default_finds_repo_root(tmp_path, monkeypatch):
    """Default behavior finds repo root and returns .corpus/index."""
    from .lib import get_index_dir

    repo_root = tmp_path / "fake_repo_index"
    repo_root.mkdir()
    (repo_root / ".gitignore").touch()

    subdir = repo_root / "src" / "submodule"
    subdir.mkdir(parents=True)

    monkeypatch.chdir(subdir)
    monkeypatch.delenv("CORPUS_INDEX_DIR", raising=False)

    result = get_index_dir()

    expected = repo_root / ".corpus" / "index"
    assert result == expected
