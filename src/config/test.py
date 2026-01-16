"""Tests for configuration management."""

import re

import pytest

from .lib import (
    EnvConfig,
    EnvVar,
    _find_repo_root,
    get_data_dir,
    get_docker_ports,
    get_environment,
    get_environment_info,
    get_index_dir,
    get_kroki_url,
    get_models_dir,
    list_environment_variables,
)

# =============================================================================
# Tests for get_environment (main interface)
# =============================================================================


class TestGetEnvironment:
    """Tests for the unified get_environment interface."""

    @pytest.mark.unit
    def test_returns_default_when_not_set(self, monkeypatch):
        """Returns default value when env var is not set."""
        monkeypatch.delenv("KROKI_PORT", raising=False)
        result = get_environment(EnvVar.KROKI_PORT)
        assert result == 18000

    @pytest.mark.unit
    def test_override_takes_priority(self, monkeypatch):
        """Override parameter takes highest priority."""
        monkeypatch.setenv("KROKI_PORT", "9999")
        result = get_environment(EnvVar.KROKI_PORT, override=5000)
        assert result == 5000

    @pytest.mark.unit
    def test_env_var_overrides_default(self, monkeypatch):
        """Environment variable overrides default value."""
        monkeypatch.setenv("KROKI_PORT", "12345")
        result = get_environment(EnvVar.KROKI_PORT)
        assert result == 12345

    @pytest.mark.unit
    def test_int_type_conversion(self, monkeypatch):
        """Integer type conversion from string."""
        monkeypatch.setenv("MCP_PORT", "8080")
        result = get_environment(EnvVar.MCP_PORT)
        assert result == 8080
        assert isinstance(result, int)

    @pytest.mark.unit
    def test_bool_type_conversion_true(self, monkeypatch):
        """Boolean type conversion for true values."""
        for value in ("true", "1", "yes", "TRUE", "Yes"):
            monkeypatch.setenv("VECTOR_USE_GPU", value)
            result = get_environment(EnvVar.VECTOR_USE_GPU)
            assert result is True

    @pytest.mark.unit
    def test_bool_type_conversion_false(self, monkeypatch):
        """Boolean type conversion for false values."""
        for value in ("false", "0", "no", "FALSE", "No"):
            monkeypatch.setenv("VECTOR_USE_GPU", value)
            result = get_environment(EnvVar.VECTOR_USE_GPU)
            assert result is False

    @pytest.mark.unit
    def test_string_type(self, monkeypatch):
        """String type returns as-is."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        result = get_environment(EnvVar.OPENAI_API_KEY)
        assert result == "sk-test-key"
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_none_default_for_api_keys(self, monkeypatch):
        """API keys default to None when not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = get_environment(EnvVar.OPENAI_API_KEY)
        assert result is None

    @pytest.mark.unit
    def test_invalid_int_returns_default(self, monkeypatch):
        """Invalid integer value returns default."""
        monkeypatch.setenv("KROKI_PORT", "not-a-number")
        result = get_environment(EnvVar.KROKI_PORT)
        assert result == 18000


class TestGetEnvironmentInfo:
    """Tests for environment variable metadata."""

    @pytest.mark.unit
    def test_returns_env_config(self):
        """Returns EnvConfig dataclass."""
        info = get_environment_info(EnvVar.KROKI_PORT)
        assert isinstance(info, EnvConfig)
        assert info.name == "KROKI_PORT"
        assert info.default == 18000
        assert info.var_type is int
        assert info.category == "docker"

    @pytest.mark.unit
    def test_description_present(self):
        """Description field is populated."""
        info = get_environment_info(EnvVar.OPENAI_API_KEY)
        assert "OpenAI" in info.description


class TestListEnvironmentVariables:
    """Tests for listing environment variables."""

    @pytest.mark.unit
    def test_returns_all_variables(self):
        """Returns all EnvVar members when no category."""
        result = list_environment_variables()
        assert len(result) == len(EnvVar)
        assert all(isinstance(v, EnvVar) for v in result)

    @pytest.mark.unit
    def test_filter_by_category(self):
        """Filters by category correctly."""
        docker_vars = list_environment_variables("docker")
        assert EnvVar.KROKI_PORT in docker_vars
        assert EnvVar.MCP_PORT in docker_vars
        assert EnvVar.OPENAI_API_KEY not in docker_vars

    @pytest.mark.unit
    def test_llm_category(self):
        """LLM category includes API keys."""
        llm_vars = list_environment_variables("llm")
        assert EnvVar.OPENAI_API_KEY in llm_vars
        assert EnvVar.ANTHROPIC_API_KEY in llm_vars


class TestGetDockerPorts:
    """Tests for Docker port configuration."""

    @pytest.mark.unit
    def test_returns_port_dict(self, monkeypatch):
        """Returns dictionary of port mappings."""
        monkeypatch.delenv("KROKI_PORT", raising=False)
        monkeypatch.delenv("MCP_PORT", raising=False)
        result = get_docker_ports()
        assert result == {"kroki": 18000, "mcp": 18080}

    @pytest.mark.unit
    def test_respects_env_overrides(self, monkeypatch):
        """Uses environment variable values."""
        monkeypatch.setenv("KROKI_PORT", "9000")
        monkeypatch.setenv("MCP_PORT", "9001")
        result = get_docker_ports()
        assert result == {"kroki": 9000, "mcp": 9001}


# =============================================================================
# Tests for get_kroki_url (convenience function)
# =============================================================================


class TestGetKrokiUrl:
    """Tests for Kroki URL resolution."""

    @pytest.mark.unit
    def test_override_takes_priority(self, monkeypatch):
        """Override parameter takes highest priority."""
        monkeypatch.setenv("KROKI_URL", "http://other:8000")
        result = get_kroki_url(override="http://custom:9000")
        assert result == "http://custom:9000"

    @pytest.mark.unit
    def test_env_var_used(self, monkeypatch):
        """KROKI_URL env var used when set."""
        monkeypatch.setenv("KROKI_URL", "http://kroki.example.com")
        result = get_kroki_url()
        assert result == "http://kroki.example.com"

    @pytest.mark.unit
    def test_computed_from_port(self, monkeypatch):
        """URL computed from KROKI_PORT when KROKI_URL not set."""
        monkeypatch.delenv("KROKI_URL", raising=False)
        monkeypatch.setenv("KROKI_PORT", "19000")
        result = get_kroki_url()
        assert result == "http://localhost:19000"

    @pytest.mark.unit
    def test_default_url(self, monkeypatch):
        """Default URL uses default port."""
        monkeypatch.delenv("KROKI_URL", raising=False)
        monkeypatch.delenv("KROKI_PORT", raising=False)
        result = get_kroki_url()
        assert result == "http://localhost:18000"


# =============================================================================
# Tests for corpus path functions
# =============================================================================


class TestGetDataDir:
    """Tests for corpus data directory resolution."""

    @pytest.mark.unit
    def test_override_takes_priority(self, tmp_path):
        """Override parameter takes highest priority."""
        custom_path = tmp_path / "custom_override"
        result = get_data_dir(custom_path)
        assert result == custom_path

    @pytest.mark.unit
    def test_string_override(self, tmp_path):
        """Override parameter accepts string paths."""
        custom_path = tmp_path / "custom_string"
        result = get_data_dir(str(custom_path))
        assert result == custom_path

    @pytest.mark.unit
    def test_env_var_used(self, tmp_path, monkeypatch):
        """CORPUS_DATA_DIR env var used when no override."""
        env_path = tmp_path / "from_env"
        monkeypatch.setenv("CORPUS_DATA_DIR", str(env_path))
        result = get_data_dir()
        assert result == env_path

    @pytest.mark.unit
    def test_override_beats_env_var(self, tmp_path, monkeypatch):
        """Override takes precedence over environment variable."""
        env_path = tmp_path / "from_env"
        override_path = tmp_path / "override"
        monkeypatch.setenv("CORPUS_DATA_DIR", str(env_path))
        result = get_data_dir(override_path)
        assert result == override_path

    @pytest.mark.unit
    def test_default_finds_repo_root(self, tmp_path, monkeypatch):
        """Default behavior finds repo root and returns .corpus/data."""
        repo_root = tmp_path / "fake_repo"
        repo_root.mkdir()
        (repo_root / ".gitignore").touch()
        subdir = repo_root / "src" / "submodule"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        monkeypatch.delenv("CORPUS_DATA_DIR", raising=False)

        result = get_data_dir()
        expected = repo_root / ".corpus" / "data"
        assert result == expected


class TestGetModelsDir:
    """Tests for models directory resolution."""

    @pytest.mark.unit
    def test_override_takes_priority(self, tmp_path):
        """Override parameter takes highest priority."""
        custom_path = tmp_path / "custom_models"
        result = get_models_dir(custom_path)
        assert result == custom_path

    @pytest.mark.unit
    def test_env_var_used(self, tmp_path, monkeypatch):
        """CORPUS_MODELS_DIR env var used when no override."""
        env_path = tmp_path / "models_from_env"
        monkeypatch.setenv("CORPUS_MODELS_DIR", str(env_path))
        result = get_models_dir()
        assert result == env_path


class TestGetIndexDir:
    """Tests for index directory resolution."""

    @pytest.mark.unit
    def test_override_takes_priority(self, tmp_path):
        """Override parameter takes highest priority."""
        custom_path = tmp_path / "custom_index"
        result = get_index_dir(custom_path)
        assert result == custom_path

    @pytest.mark.unit
    def test_env_var_used(self, tmp_path, monkeypatch):
        """CORPUS_INDEX_DIR env var used when no override."""
        env_path = tmp_path / "index_from_env"
        monkeypatch.setenv("CORPUS_INDEX_DIR", str(env_path))
        result = get_index_dir()
        assert result == env_path


# =============================================================================
# Tests for repository root detection
# =============================================================================


class TestFindRepoRoot:
    """Tests for repository root detection."""

    @pytest.mark.unit
    def test_from_cwd(self, tmp_path, monkeypatch):
        """Finds .gitignore from current directory."""
        repo_root = tmp_path / "test_repo"
        repo_root.mkdir()
        (repo_root / ".gitignore").touch()
        monkeypatch.chdir(repo_root)

        result = _find_repo_root()
        assert result == repo_root

    @pytest.mark.unit
    def test_from_subdirectory(self, tmp_path, monkeypatch):
        """Walks up from subdirectory."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (repo_root / ".gitignore").touch()
        deep_subdir = repo_root / "a" / "b" / "c"
        deep_subdir.mkdir(parents=True)
        monkeypatch.chdir(deep_subdir)

        result = _find_repo_root()
        assert result == repo_root

    @pytest.mark.unit
    def test_explicit_start_path(self, tmp_path):
        """Accepts explicit start path."""
        repo_root = tmp_path / "explicit_repo"
        repo_root.mkdir()
        (repo_root / ".gitignore").touch()
        subdir = repo_root / "src"
        subdir.mkdir()

        result = _find_repo_root(start_path=subdir)
        assert result == repo_root

    @pytest.mark.unit
    def test_not_found_raises_error(self, tmp_path, monkeypatch):
        """RuntimeError raised when .gitignore not found."""
        no_repo = tmp_path / "not_a_repo"
        no_repo.mkdir()
        monkeypatch.chdir(no_repo)

        with pytest.raises(RuntimeError, match="Could not find repository root"):
            _find_repo_root()

    @pytest.mark.unit
    def test_error_includes_start_path(self, tmp_path):
        """Error message includes the start path."""
        no_repo = tmp_path / "no_git_here"
        no_repo.mkdir()

        with pytest.raises(RuntimeError, match=re.escape(str(no_repo))):
            _find_repo_root(start_path=no_repo)
