"""Unit tests for health checking module."""

import pytest

from .health import (
    HealthStatus,
    ServerHealth,
    ServiceStatus,
    check_history_db,
    check_kroki,
    check_llm_providers,
    check_rag_index,
    format_startup_banner,
    get_server_health,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    @pytest.mark.unit
    def test_status_values(self):
        """Health status has expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    @pytest.mark.unit
    def test_status_is_string_enum(self):
        """HealthStatus inherits from str for JSON serialization."""
        assert isinstance(HealthStatus.HEALTHY, str)
        assert HealthStatus.HEALTHY == "healthy"


class TestServiceStatus:
    """Tests for ServiceStatus dataclass."""

    @pytest.mark.unit
    def test_available_service(self):
        """Available service has correct state."""
        status = ServiceStatus(
            available=True,
            message="Service running",
        )
        assert status.available is True
        assert status.message == "Service running"
        assert status.details == {}

    @pytest.mark.unit
    def test_unavailable_service(self):
        """Unavailable service has correct state."""
        status = ServiceStatus(
            available=False,
            message="Service not found",
        )
        assert status.available is False
        assert status.message == "Service not found"

    @pytest.mark.unit
    def test_service_with_details(self):
        """Service status can include extra details."""
        status = ServiceStatus(
            available=True,
            message="OK",
            details={"url": "http://localhost:8000", "version": "1.0"},
        )
        assert status.details["url"] == "http://localhost:8000"
        assert status.details["version"] == "1.0"


class TestCheckKroki:
    """Tests for check_kroki function."""

    @pytest.mark.unit
    def test_returns_service_status(self, monkeypatch):
        """check_kroki returns ServiceStatus."""

        class MockClient:
            base_url = "http://localhost:8000"

            def is_available(self):
                return False

        monkeypatch.setattr("src.render.RenderClient", MockClient)

        result = check_kroki()

        assert isinstance(result, ServiceStatus)
        assert result.available is False
        assert "Kroki not responding" in result.message

    @pytest.mark.unit
    def test_available_kroki(self, monkeypatch):
        """check_kroki returns available when service responds."""

        class MockClient:
            base_url = "http://localhost:8000"

            def is_available(self):
                return True

        monkeypatch.setattr("src.render.RenderClient", MockClient)

        result = check_kroki()

        assert result.available is True
        assert "running" in result.message
        assert result.details["url"] == "http://localhost:8000"

    @pytest.mark.unit
    def test_exception_handling(self, monkeypatch):
        """check_kroki handles import/runtime errors gracefully."""

        class BrokenClient:
            def __init__(self):
                raise RuntimeError("render module broken")

        monkeypatch.setattr("src.render.RenderClient", BrokenClient)

        result = check_kroki()

        assert result.available is False
        assert "failed" in result.message.lower()


class TestCheckLLMProviders:
    """Tests for check_llm_providers function."""

    @pytest.mark.unit
    def test_returns_service_status(self):
        """check_llm_providers returns ServiceStatus."""
        result = check_llm_providers()

        assert isinstance(result, ServiceStatus)
        assert isinstance(result.available, bool)

    @pytest.mark.unit
    def test_no_providers(self, monkeypatch):
        """check_llm_providers reports when no providers available."""
        monkeypatch.setattr("src.config.get_available_llm_providers", lambda: [])

        result = check_llm_providers()

        assert result.available is False
        assert "No LLM providers" in result.message
        assert result.details["available_providers"] == []

    @pytest.mark.unit
    def test_with_providers(self, monkeypatch):
        """check_llm_providers reports available providers."""
        monkeypatch.setattr(
            "src.config.get_available_llm_providers",
            lambda: ["openai", "anthropic"],
        )

        result = check_llm_providers()

        assert result.available is True
        assert "openai" in result.message
        assert "anthropic" in result.message
        assert result.details["available_providers"] == ["openai", "anthropic"]


class TestCheckRAGIndex:
    """Tests for check_rag_index function."""

    @pytest.mark.unit
    def test_returns_service_status(self):
        """check_rag_index returns ServiceStatus."""
        result = check_rag_index()

        assert isinstance(result, ServiceStatus)
        assert isinstance(result.available, bool)

    @pytest.mark.unit
    def test_missing_index(self, monkeypatch, tmp_path):
        """check_rag_index reports when index not found."""
        # Point to empty temp dir
        monkeypatch.setattr("src.config.get_index_dir", lambda: tmp_path)

        result = check_rag_index()

        assert result.available is False
        assert "not found" in result.message


class TestCheckHistoryDb:
    """Tests for check_history_db function."""

    @pytest.mark.unit
    def test_returns_service_status(self):
        """check_history_db returns ServiceStatus."""
        result = check_history_db()

        assert isinstance(result, ServiceStatus)
        assert isinstance(result.available, bool)


class TestGetServerHealth:
    """Tests for get_server_health function."""

    @pytest.mark.unit
    def test_returns_server_health(self):
        """get_server_health returns ServerHealth instance."""
        health = get_server_health()

        assert isinstance(health, ServerHealth)

    @pytest.mark.unit
    def test_returns_complete_health(self):
        """get_server_health returns all required fields."""
        health = get_server_health()

        assert hasattr(health, "status")
        assert hasattr(health, "can_generate")
        assert hasattr(health, "can_preview")
        assert hasattr(health, "can_use_rag")
        assert hasattr(health, "version")
        assert hasattr(health, "checked_at")

    @pytest.mark.unit
    def test_service_statuses(self):
        """get_server_health includes all service statuses."""
        health = get_server_health()

        assert isinstance(health.kroki, ServiceStatus)
        assert isinstance(health.rag_index, ServiceStatus)
        assert isinstance(health.llm_providers, ServiceStatus)
        assert isinstance(health.history_db, ServiceStatus)

    @pytest.mark.unit
    def test_to_dict_format(self):
        """Health converts to expected dict format."""
        health = get_server_health()
        result = health.to_dict()

        assert "status" in result
        assert "version" in result
        assert "checked_at" in result
        assert "capabilities" in result
        assert "services" in result

    @pytest.mark.unit
    def test_to_dict_services_structure(self):
        """to_dict includes all services with correct structure."""
        health = get_server_health()
        result = health.to_dict()

        services = result["services"]
        assert "kroki" in services
        assert "rag_index" in services
        assert "llm_providers" in services
        assert "history_db" in services

        # Each service has expected fields
        for service_name, service_data in services.items():
            assert "available" in service_data, f"{service_name} missing 'available'"
            assert "message" in service_data, f"{service_name} missing 'message'"

    @pytest.mark.unit
    def test_to_dict_capabilities_structure(self):
        """to_dict includes all capabilities."""
        health = get_server_health()
        result = health.to_dict()

        capabilities = result["capabilities"]
        assert "generate_layout" in capabilities
        assert "preview_layout" in capabilities
        assert "generate_variations" in capabilities
        assert "rag_context" in capabilities

    @pytest.mark.unit
    def test_capabilities_match_services(self):
        """Capabilities reflect actual service status."""
        health = get_server_health()
        result = health.to_dict()

        # Capabilities should match service availability
        assert (
            result["capabilities"]["preview_layout"]
            == result["services"]["kroki"]["available"]
        )
        assert (
            result["capabilities"]["rag_context"]
            == result["services"]["rag_index"]["available"]
        )
        assert (
            result["capabilities"]["generate_layout"]
            == result["services"]["llm_providers"]["available"]
        )


class TestFormatStartupBanner:
    """Tests for format_startup_banner function."""

    @pytest.mark.unit
    def test_returns_string(self):
        """format_startup_banner returns a string."""
        health = get_server_health()
        banner = format_startup_banner(health)

        assert isinstance(banner, str)

    @pytest.mark.unit
    def test_includes_version(self):
        """Banner includes server version."""
        health = get_server_health()
        banner = format_startup_banner(health)

        assert health.version in banner

    @pytest.mark.unit
    def test_includes_status(self):
        """Banner includes health status."""
        health = get_server_health()
        banner = format_startup_banner(health)

        assert health.status.value.upper() in banner

    @pytest.mark.unit
    def test_includes_services_section(self):
        """Banner includes services section."""
        health = get_server_health()
        banner = format_startup_banner(health)

        assert "Services:" in banner
        assert "LLM Providers" in banner
        assert "Kroki" in banner
        assert "RAG Index" in banner
        assert "History DB" in banner

    @pytest.mark.unit
    def test_includes_capabilities_section(self):
        """Banner includes capabilities section."""
        health = get_server_health()
        banner = format_startup_banner(health)

        assert "Capabilities:" in banner
        assert "generate_layout" in banner
        assert "preview_layout" in banner
        assert "rag_context" in banner
