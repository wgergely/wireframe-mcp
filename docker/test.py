"""Tests for Docker module."""

from docker import CONFIG_PATH, CORPUS_DATA_PATH, DOCKER_IMAGE_NAME, DOCKER_REGISTRY
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
        assert str(CORPUS_DATA_PATH) == "/app/corpus"

    def test_config_path(self) -> None:
        """Config path is a valid POSIX path."""
        assert str(CONFIG_PATH) == "/app/config"


class TestGetContainerPath:
    """Test container path utility function."""

    def test_get_container_path_default_base(self) -> None:
        """Construct path with default base."""
        result = get_container_path("layouts/v1")
        assert result == "/app/corpus/layouts/v1"

    def test_get_container_path_custom_base(self) -> None:
        """Construct path with custom base."""
        result = get_container_path("settings.json", base=CONFIG_PATH)
        assert result == "/app/config/settings.json"

    def test_get_container_path_nested(self) -> None:
        """Construct nested paths correctly."""
        result = get_container_path("data/models/v2/index.json")
        assert result == "/app/corpus/data/models/v2/index.json"
