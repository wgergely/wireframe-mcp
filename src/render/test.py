"""Tests for render module.

Unit tests are mocked (no network).
Integration tests require a running Kroki container.
"""

import base64
import zlib
from dataclasses import dataclass

import httpx
import pytest

from src.render import (
    D2Theme,
    OutputFormat,
    PlantUMLTheme,
    RenderClient,
    RenderConfig,
    RenderError,
    RenderOutput,
)

# =============================================================================
# Unit Tests (Mocked)
# =============================================================================


class TestRenderConfig:
    """Tests for RenderConfig dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Default config has sensible values."""
        config = RenderConfig()
        assert config.output_format == OutputFormat.PNG
        assert config.theme is None
        assert config.scale == 1.0
        assert config.width is None
        assert config.height is None

    @pytest.mark.unit
    def test_custom_values(self):
        """Config accepts custom values."""
        config = RenderConfig(
            output_format=OutputFormat.SVG,
            theme=D2Theme.TERMINAL,
            scale=2.0,
            width=800,
            height=600,
        )
        assert config.output_format == OutputFormat.SVG
        assert config.theme == D2Theme.TERMINAL
        assert config.scale == 2.0

    @pytest.mark.unit
    def test_invalid_scale_low(self):
        """Config rejects scale below minimum."""
        with pytest.raises(ValueError, match="Scale must be between"):
            RenderConfig(scale=0.05)

    @pytest.mark.unit
    def test_invalid_scale_high(self):
        """Config rejects scale above maximum."""
        with pytest.raises(ValueError, match="Scale must be between"):
            RenderConfig(scale=10.0)

    @pytest.mark.unit
    def test_invalid_width(self):
        """Config rejects negative width."""
        with pytest.raises(ValueError, match="Width must be positive"):
            RenderConfig(width=-100)

    @pytest.mark.unit
    def test_invalid_height(self):
        """Config rejects zero height."""
        with pytest.raises(ValueError, match="Height must be positive"):
            RenderConfig(height=0)


class TestRenderOutput:
    """Tests for RenderOutput dataclass."""

    @pytest.mark.unit
    def test_size_bytes(self):
        """Size property returns correct byte count."""
        output = RenderOutput(
            image_bytes=b"test image data",
            format=OutputFormat.PNG,
            diagram_type="d2",
        )
        assert output.size_bytes == 15

    @pytest.mark.unit
    def test_save(self, tmp_path):
        """Save writes image to file."""
        output = RenderOutput(
            image_bytes=b"\x89PNG\r\n\x1a\nfake",
            format=OutputFormat.PNG,
            diagram_type="d2",
        )
        path = tmp_path / "test.png"
        output.save(str(path))
        assert path.read_bytes() == b"\x89PNG\r\n\x1a\nfake"


class TestRenderClientEncoding:
    """Tests for diagram encoding."""

    @pytest.mark.unit
    def test_encode_diagram(self):
        """Encode produces valid base64 URL-safe string."""
        client = RenderClient()
        encoded = client._encode_diagram("test diagram")

        # Should be URL-safe base64 (only alphanumeric, - and _)
        assert all(c.isalnum() or c in "-_=" for c in encoded)

        # Should be decodable back to original
        decoded = zlib.decompress(base64.urlsafe_b64decode(encoded))
        assert decoded == b"test diagram"


class TestD2OptionInjection:
    """Tests for D2 option injection."""

    @pytest.mark.unit
    def test_inject_theme(self):
        """Theme is injected as d2-config comment."""
        client = RenderClient()
        config = RenderConfig(theme=D2Theme.TERMINAL)
        result = client._inject_d2_options("root: Test { }", config)
        assert "# d2-config: --theme 300" in result
        assert "root: Test { }" in result

    @pytest.mark.unit
    def test_inject_scale(self):
        """Scale is injected as d2-config comment."""
        client = RenderClient()
        config = RenderConfig(scale=2.0)
        result = client._inject_d2_options("root: Test { }", config)
        assert "# d2-config: --scale 2.0" in result

    @pytest.mark.unit
    def test_inject_sketch(self):
        """Sketch mode is injected."""
        client = RenderClient()
        config = RenderConfig(sketch=True)
        result = client._inject_d2_options("root: Test { }", config)
        assert "# d2-config: --sketch" in result

    @pytest.mark.unit
    def test_no_injection_for_defaults(self):
        """Default config doesn't inject options."""
        client = RenderClient()
        config = RenderConfig()
        result = client._inject_d2_options("root: Test { }", config)
        assert result == "root: Test { }"


class TestPlantUMLOptionInjection:
    """Tests for PlantUML option injection."""

    @pytest.mark.unit
    def test_inject_theme(self):
        """Theme is injected after @startsalt."""
        client = RenderClient()
        config = RenderConfig(theme=PlantUMLTheme.SKETCHY)
        dsl = "@startsalt\n{\n  [Button]\n}\n@endsalt"
        result = client._inject_plantuml_options(dsl, config)
        assert "!theme sketchy" in result
        # Theme should come after @startsalt
        lines = result.split("\n")
        start_idx = next(i for i, line in enumerate(lines) if "@startsalt" in line)
        theme_idx = next(i for i, line in enumerate(lines) if "!theme" in line)
        assert theme_idx > start_idx

    @pytest.mark.unit
    def test_inject_scale(self):
        """Scale is injected after @startsalt."""
        client = RenderClient()
        config = RenderConfig(scale=1.5)
        dsl = "@startsalt\n{\n  [Button]\n}\n@endsalt"
        result = client._inject_plantuml_options(dsl, config)
        assert "scale 1.5" in result


@dataclass
class MockResponse:
    """Mock HTTP response for testing."""

    status_code: int
    content: bytes = b""
    text: str = ""


class TestRenderClientMocked:
    """Tests for RenderClient with mocked HTTP."""

    @pytest.mark.unit
    def test_is_available_success(self, monkeypatch):
        """is_available returns True on 200 response."""
        client = RenderClient()

        def mock_get(*args, **kwargs):
            return MockResponse(status_code=200)

        monkeypatch.setattr(client._client, "get", mock_get)
        assert client.is_available() is True

    @pytest.mark.unit
    def test_is_available_failure(self, monkeypatch):
        """is_available returns False on connection error."""
        client = RenderClient()

        def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(client._client, "get", mock_get)
        assert client.is_available() is False

    @pytest.mark.unit
    def test_render_success(self, monkeypatch):
        """render returns RenderOutput on success."""
        client = RenderClient()
        fake_png = b"\x89PNG\r\n\x1a\nfake"

        def mock_get(*args, **kwargs):
            return MockResponse(status_code=200, content=fake_png)

        monkeypatch.setattr(client._client, "get", mock_get)

        result = client.render("root: Test { }", "d2")

        assert result.image_bytes == fake_png
        assert result.format == OutputFormat.PNG
        assert result.diagram_type == "d2"

    @pytest.mark.unit
    def test_render_error_status(self, monkeypatch):
        """render raises RenderError on non-200 status."""
        client = RenderClient()

        def mock_get(*args, **kwargs):
            return MockResponse(status_code=400, text="Bad Request")

        monkeypatch.setattr(client._client, "get", mock_get)

        with pytest.raises(RenderError) as exc_info:
            client.render("invalid { }", "d2")
        assert exc_info.value.status_code == 400

    @pytest.mark.unit
    def test_render_unsupported_type(self):
        """render raises RenderError for unsupported diagram type."""
        client = RenderClient()
        with pytest.raises(RenderError, match="Unsupported diagram type"):
            client.render("test", "mermaid")  # type: ignore


# =============================================================================
# Integration Tests (require running Kroki)
# =============================================================================


@pytest.fixture
def kroki_client():
    """Create a RenderClient for integration tests."""
    client = RenderClient()
    if not client.is_available():
        pytest.skip("Kroki service not available")
    return client


class TestRenderClientIntegration:
    """Integration tests requiring a running Kroki container."""

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_render_d2_default(self, kroki_client):
        """Render D2 diagram to SVG (Kroki D2 only supports SVG output)."""
        dsl = """
        root: Dashboard {
          direction: right
          sidebar: Navigation { }
          main: Content { }
        }
        """
        config = RenderConfig(output_format=OutputFormat.SVG)
        result = kroki_client.render(dsl, "d2", config)

        assert b"<svg" in result.image_bytes
        assert result.format == OutputFormat.SVG
        assert result.size_bytes > 100

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_render_d2_svg(self, kroki_client):
        """Render D2 diagram to SVG."""
        config = RenderConfig(output_format=OutputFormat.SVG)
        result = kroki_client.render("root: Test { }", "d2", config)

        assert b"<svg" in result.image_bytes
        assert result.format == OutputFormat.SVG

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_render_d2_with_theme(self, kroki_client):
        """Render D2 diagram with terminal theme (SVG output)."""
        config = RenderConfig(
            output_format=OutputFormat.SVG,
            theme=D2Theme.TERMINAL,
        )
        result = kroki_client.render("root: Test { }", "d2", config)
        assert b"<svg" in result.image_bytes

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_render_plantuml_png(self, kroki_client):
        """Render PlantUML Salt diagram to PNG."""
        dsl = """
        @startsalt
        {
          "Dashboard"
          { [Button 1] | [Button 2] }
        }
        @endsalt
        """
        result = kroki_client.render(dsl, "plantuml")
        assert result.image_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_render_plantuml_with_theme(self, kroki_client):
        """Render PlantUML with sketchy theme."""
        config = RenderConfig(
            output_format=OutputFormat.PNG,
            theme=PlantUMLTheme.SKETCHY,
        )
        dsl = "@startsalt\n{\n  [Button]\n}\n@endsalt"
        result = kroki_client.render(dsl, "plantuml", config)
        assert result.image_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_render_layout_d2(self, kroki_client):
        """Render LayoutNode via D2 provider (SVG output)."""
        from src.mid import ComponentType, LayoutNode, Orientation

        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            label="Dashboard",
            orientation=Orientation.HORIZONTAL,
            children=[
                LayoutNode(id="sidebar", type=ComponentType.DRAWER, flex_ratio=3),
                LayoutNode(id="main", type=ComponentType.CONTAINER, flex_ratio=9),
            ],
        )

        config = RenderConfig(output_format=OutputFormat.SVG)
        result = kroki_client.render_layout(node, "d2", config)
        assert b"<svg" in result.image_bytes
        assert result.size_bytes > 100

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_render_layout_plantuml(self, kroki_client):
        """Render LayoutNode via PlantUML provider."""
        from src.mid import ComponentType, LayoutNode

        node = LayoutNode(
            id="card",
            type=ComponentType.CARD,
            label="My Card",
            children=[
                LayoutNode(id="title", type=ComponentType.TEXT, label="Title"),
                LayoutNode(id="btn", type=ComponentType.BUTTON, label="Click"),
            ],
        )

        result = kroki_client.render_layout(node, "plantuml")
        assert result.image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
