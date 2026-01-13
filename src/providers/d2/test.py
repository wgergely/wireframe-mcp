"""Unit tests for D2 provider."""

import pytest

from src.ir import ComponentType, LayoutNode, Orientation
from src.providers.d2 import D2Provider


@pytest.fixture
def provider():
    """Create a D2Provider instance."""
    return D2Provider()


class TestD2Provider:
    """Tests for D2Provider."""

    @pytest.mark.unit
    def test_provider_name(self, provider):
        """Provider has correct name."""
        assert provider.name == "d2"

    @pytest.mark.unit
    def test_file_extension(self, provider):
        """Provider has correct file extension."""
        assert provider.file_extension == ".d2"

    @pytest.mark.unit
    def test_simple_container(self, provider):
        """Single node produces valid D2 block."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            label="Main",
        )
        result = provider.transpile(node)
        assert "root: Main {" in result
        assert result.strip().endswith("}")

    @pytest.mark.unit
    def test_horizontal_direction(self, provider):
        """Horizontal orientation adds direction: right."""
        node = LayoutNode(
            id="row",
            type=ComponentType.CONTAINER,
            orientation=Orientation.HORIZONTAL,
        )
        result = provider.transpile(node)
        assert "direction: right" in result

    @pytest.mark.unit
    def test_vertical_no_direction(self, provider):
        """Vertical orientation does not add direction."""
        node = LayoutNode(
            id="col",
            type=ComponentType.CONTAINER,
            orientation=Orientation.VERTICAL,
        )
        result = provider.transpile(node)
        assert "direction:" not in result

    @pytest.mark.unit
    def test_nested_containers(self, provider):
        """Multi-level hierarchy transpiles correctly."""
        child = LayoutNode(
            id="child",
            type=ComponentType.BUTTON,
            label="Click",
        )
        parent = LayoutNode(
            id="parent",
            type=ComponentType.CONTAINER,
            children=[child],
        )
        result = provider.transpile(parent)
        assert "parent: container {" in result
        assert "child: Click {" in result
        # Proper nesting (child is indented)
        lines = result.split("\n")
        parent_indent = next(
            len(line) - len(line.lstrip()) for line in lines if "parent:" in line
        )
        child_indent = next(
            len(line) - len(line.lstrip()) for line in lines if "child:" in line
        )
        assert child_indent > parent_indent

    @pytest.mark.unit
    def test_flex_ratio_width(self, provider):
        """Flex ratio converts to width percentage."""
        node = LayoutNode(
            id="sidebar",
            type=ComponentType.DRAWER,
            flex_ratio=3,
        )
        result = provider.transpile(node)
        assert "width: 25%" in result  # 3/12 = 25%

    @pytest.mark.unit
    def test_flex_ratio_default_no_width(self, provider):
        """Default flex ratio (1) does not add width."""
        node = LayoutNode(
            id="item",
            type=ComponentType.CONTAINER,
            flex_ratio=1,
        )
        result = provider.transpile(node)
        assert "width:" not in result

    @pytest.mark.unit
    def test_label_escaping(self, provider):
        """Special characters in labels are escaped."""
        node = LayoutNode(
            id="test",
            type=ComponentType.TEXT,
            label="Title: Subtitle",
        )
        result = provider.transpile(node)
        # Colon should trigger quoting
        assert '"Title: Subtitle"' in result

    @pytest.mark.unit
    def test_no_label_uses_type(self, provider):
        """Node without label uses component type."""
        node = LayoutNode(
            id="btn",
            type=ComponentType.BUTTON,
        )
        result = provider.transpile(node)
        assert "btn: button {" in result

    @pytest.mark.unit
    def test_complex_layout(self, provider):
        """Dashboard layout with sidebar and main content."""
        layout = LayoutNode(
            id="dashboard",
            type=ComponentType.CONTAINER,
            orientation=Orientation.HORIZONTAL,
            children=[
                LayoutNode(
                    id="sidebar",
                    type=ComponentType.DRAWER,
                    label="Navigation",
                    flex_ratio=3,
                ),
                LayoutNode(
                    id="main",
                    type=ComponentType.CONTAINER,
                    label="Content",
                    flex_ratio=9,
                ),
            ],
        )
        result = provider.transpile(layout)
        assert "dashboard: container {" in result
        assert "direction: right" in result
        assert "sidebar: Navigation {" in result
        assert "width: 25%" in result  # 3/12
        assert "main: Content {" in result
        assert "width: 75%" in result  # 9/12
