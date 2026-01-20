"""Tests for Docker module."""

import pytest

from docker import (
    CONFIG_PATH,
    CORPUS_DATA_PATH,
    DOCKER_IMAGE_NAME,
    DOCKER_REGISTRY,
    NETWORK_NAME,
    PROJECT_NAME,
    PROJECT_PREFIX,
    Service,
    ServiceCategory,
    get_compose_files,
    get_service_info,
    list_backends,
    list_modes,
    list_services,
    list_services_by_category,
)
from docker.lib import get_container_path


@pytest.mark.unit
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

    def test_project_prefix(self) -> None:
        """Project prefix follows convention."""
        assert PROJECT_PREFIX == "wfmcp"

    def test_project_name(self) -> None:
        """Project name is correctly set."""
        assert PROJECT_NAME == "wireframe-mcp"

    def test_network_name(self) -> None:
        """Network name uses project prefix."""
        assert NETWORK_NAME == "wfmcp-network"


@pytest.mark.unit
class TestServiceEnums:
    """Test Docker service enumeration."""

    def test_service_enum_has_server(self) -> None:
        """Service enum includes SERVER."""
        assert Service.SERVER is not None

    def test_service_enum_has_kroki(self) -> None:
        """Service enum includes KROKI."""
        assert Service.KROKI is not None

    def test_get_service_info_returns_data(self) -> None:
        """get_service_info returns ServiceInfo with expected fields."""
        info = get_service_info(Service.SERVER)
        assert info.container_name == "wfmcp-server"
        assert info.category == ServiceCategory.CORE
        assert info.internal_port == 18080

    def test_kroki_service_info(self) -> None:
        """Kroki service has correct metadata."""
        info = get_service_info(Service.KROKI)
        assert info.container_name == "wfmcp-kroki"
        assert info.hostname == "kroki"
        assert info.category == ServiceCategory.RENDERER

    def test_list_services_returns_all(self) -> None:
        """list_services returns all services."""
        services = list_services()
        assert len(services) == len(Service)

    def test_list_services_by_category(self) -> None:
        """list_services filters by category."""
        core_services = list_services(ServiceCategory.CORE)
        assert Service.SERVER in core_services
        assert Service.KROKI not in core_services

    def test_list_services_by_category_dict(self) -> None:
        """list_services_by_category returns dict of all categories."""
        grouped = list_services_by_category()
        assert ServiceCategory.CORE in grouped
        assert ServiceCategory.RENDERER in grouped
        assert Service.SERVER in grouped[ServiceCategory.CORE]


@pytest.mark.unit
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


@pytest.mark.unit
class TestListModes:
    """Test available docker-compose modes."""

    def test_list_modes_returns_list(self) -> None:
        """list_modes returns a list."""
        modes = list_modes()
        assert isinstance(modes, list)

    def test_list_modes_includes_dev_and_production(self) -> None:
        """list_modes includes dev, hotreload, and production modes."""
        modes = list_modes()
        assert "dev" in modes  # backwards compat alias for hotreload
        assert "hotreload" in modes
        assert "production" in modes

    def test_list_modes_sorted(self) -> None:
        """list_modes returns sorted list."""
        modes = list_modes()
        assert modes == sorted(modes)


@pytest.mark.unit
class TestListBackends:
    """Test available docker-compose backends/services."""

    def test_list_backends_returns_list(self) -> None:
        """list_backends returns a list."""
        backends = list_backends()
        assert isinstance(backends, list)

    def test_list_backends_includes_core_and_renderer(self) -> None:
        """list_backends includes core and renderer service files."""
        backends = list_backends()
        assert "core" in backends
        assert "renderer" in backends

    def test_list_backends_sorted(self) -> None:
        """list_backends returns sorted list."""
        backends = list_backends()
        assert backends == sorted(backends)


@pytest.mark.unit
class TestGetComposeFiles:
    """Test compose file composition logic."""

    def test_get_compose_files_dev_mode(self) -> None:
        """Get compose files for dev mode."""
        files = get_compose_files(mode="dev")
        assert isinstance(files, list)
        assert len(files) >= 1
        # Base file should be first
        assert files[0].name == "compose.base.yml"

    def test_get_compose_files_prod_mode(self) -> None:
        """Get compose files for prod mode."""
        files = get_compose_files(mode="production")
        assert isinstance(files, list)
        assert len(files) >= 1
        assert files[0].name == "compose.base.yml"

    def test_get_compose_files_hotreload_mode(self) -> None:
        """Get compose files for hotreload mode."""
        files = get_compose_files(mode="hotreload")
        file_names = [f.name for f in files]
        assert "compose.base.yml" in file_names
        assert "compose.hotreload.yml" in file_names

    def test_get_compose_files_invalid_mode(self) -> None:
        """Invalid mode raises ValueError."""
        try:
            get_compose_files(mode="invalid")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Invalid mode" in str(e)

    def test_get_compose_files_includes_kroki_by_default(self) -> None:
        """Kroki is included by default."""
        files = get_compose_files(mode="dev")
        file_names = [f.name for f in files]
        assert "compose.renderer.yml" in file_names

    def test_get_compose_files_without_kroki(self) -> None:
        """Kroki can be excluded."""
        files = get_compose_files(mode="dev", include_kroki=False)
        file_names = [f.name for f in files]
        assert "compose.renderer.yml" not in file_names

    def test_get_compose_files_includes_server_by_default(self) -> None:
        """Server is included by default."""
        files = get_compose_files(mode="dev")
        file_names = [f.name for f in files]
        assert "compose.core.yml" in file_names

    def test_get_compose_files_without_server(self) -> None:
        """Server can be excluded."""
        files = get_compose_files(mode="dev", include_server=False)
        file_names = [f.name for f in files]
        assert "compose.core.yml" not in file_names

    def test_get_compose_files_order(self) -> None:
        """Compose files are in correct order (base, server, kroki, mode)."""
        files = get_compose_files(mode="dev", include_kroki=True)
        file_names = [f.name for f in files]
        # Order: base, server, kroki, mode overlay
        base_idx = file_names.index("compose.base.yml")
        server_idx = file_names.index("compose.core.yml")
        kroki_idx = file_names.index("compose.renderer.yml")
        hotreload_idx = file_names.index("compose.hotreload.yml")
        assert base_idx < server_idx < kroki_idx < hotreload_idx
