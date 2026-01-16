"""Tests for Docker module."""

from docker import (
    CONFIG_PATH,
    CORPUS_DATA_PATH,
    DOCKER_IMAGE_NAME,
    DOCKER_REGISTRY,
    get_compose_files,
    list_backends,
    list_modes,
)
from docker.lib import get_container_path


class TestDockerConstants:
    """Test Docker configuration constants."""

    def test_docker_image_name(self) -> None:
        """Docker image name is correctly set."""
        assert DOCKER_IMAGE_NAME == "wireframe-mcp"

    def test_docker_registry(self) -> None:
        """Docker registry is correctly set."""
        assert DOCKER_REGISTRY == "ghcr.io/wgergely"

    def test_corpus_data_path(self) -> None:
        """Corpus data path is a valid POSIX path."""
        assert str(CORPUS_DATA_PATH) == "/app/corpus/data"

    def test_config_path(self) -> None:
        """Config path is a valid POSIX path."""
        assert str(CONFIG_PATH) == "/app/config"


class TestGetContainerPath:
    """Test container path utility function."""

    def test_get_container_path_default_base(self) -> None:
        """Construct path with default base."""
        result = get_container_path("layouts/v1")
        assert result == "/app/corpus/data/layouts/v1"

    def test_get_container_path_custom_base(self) -> None:
        """Construct path with custom base."""
        result = get_container_path("settings.json", base=CONFIG_PATH)
        assert result == "/app/config/settings.json"

    def test_get_container_path_nested(self) -> None:
        """Construct nested paths correctly."""
        result = get_container_path("models/v2/index.json")
        assert result == "/app/corpus/data/models/v2/index.json"


class TestListModes:
    """Test available docker-compose modes."""

    def test_list_modes_returns_list(self) -> None:
        """list_modes returns a list."""
        modes = list_modes()
        assert isinstance(modes, list)

    def test_list_modes_includes_dev_and_prod(self) -> None:
        """list_modes includes dev and prod modes."""
        modes = list_modes()
        assert "dev" in modes
        assert "prod" in modes

    def test_list_modes_sorted(self) -> None:
        """list_modes returns sorted list."""
        modes = list_modes()
        assert modes == sorted(modes)


class TestListBackends:
    """Test available docker-compose backends."""

    def test_list_backends_returns_list(self) -> None:
        """list_backends returns a list."""
        backends = list_backends()
        assert isinstance(backends, list)

    def test_list_backends_includes_kroki(self) -> None:
        """list_backends includes kroki backend."""
        backends = list_backends()
        assert "kroki" in backends

    def test_list_backends_sorted(self) -> None:
        """list_backends returns sorted list."""
        backends = list_backends()
        assert backends == sorted(backends)


class TestGetComposeFiles:
    """Test compose file composition logic."""

    def test_get_compose_files_dev_mode(self) -> None:
        """Get compose files for dev mode."""
        files = get_compose_files(mode="dev")
        assert isinstance(files, list)
        assert len(files) >= 1
        # Base file should be first
        assert files[0].name == "docker-compose.yml"

    def test_get_compose_files_prod_mode(self) -> None:
        """Get compose files for prod mode."""
        files = get_compose_files(mode="prod")
        assert isinstance(files, list)
        assert len(files) >= 1
        assert files[0].name == "docker-compose.yml"

    def test_get_compose_files_invalid_mode(self) -> None:
        """Invalid mode raises ValueError."""
        try:
            get_compose_files(mode="invalid")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Invalid mode" in str(e)

    def test_get_compose_files_with_kroki(self) -> None:
        """Include kroki backend in compose files."""
        files = get_compose_files(mode="dev", include_kroki=True)
        file_names = [f.name for f in files]
        assert "docker-compose.yml" in file_names
        assert "docker-compose.kroki.yml" in file_names

    def test_get_compose_files_with_backends(self) -> None:
        """Include additional backends in compose files."""
        files = get_compose_files(mode="dev", include_backends=["kroki"])
        file_names = [f.name for f in files]
        assert "docker-compose.yml" in file_names
        assert "docker-compose.kroki.yml" in file_names

    def test_get_compose_files_order(self) -> None:
        """Compose files are in correct order (base, mode, kroki, backends)."""
        files = get_compose_files(mode="dev", include_kroki=True)
        file_names = [f.name for f in files]
        # Base should come before mode-specific
        base_idx = file_names.index("docker-compose.yml")
        dev_idx = file_names.index("docker-compose.dev.yml")
        kroki_idx = file_names.index("docker-compose.kroki.yml")
        assert base_idx < dev_idx < kroki_idx
