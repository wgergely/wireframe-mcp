"""Unit tests for PlantUML Salt provider."""

import pytest

from src.ir import ComponentType, LayoutNode, Orientation
from src.providers.plantuml import PlantUMLProvider


@pytest.fixture
def provider():
    """Create a PlantUMLProvider instance."""
    return PlantUMLProvider()


class TestPlantUMLProvider:
    """Tests for PlantUMLProvider."""

    @pytest.mark.unit
    def test_provider_name(self, provider):
        """Provider has correct name."""
        assert provider.name == "plantuml"

    @pytest.mark.unit
    def test_file_extension(self, provider):
        """Provider has correct file extension."""
        assert provider.file_extension == ".puml"

    @pytest.mark.unit
    def test_salt_wrapper(self, provider):
        """Output is wrapped in @startsalt/@endsalt."""
        node = LayoutNode(id="root", type=ComponentType.CONTAINER)
        result = provider.transpile(node)
        assert result.startswith("@startsalt")
        assert result.endswith("@endsalt")

    @pytest.mark.unit
    def test_button_syntax(self, provider):
        """Button renders with square brackets."""
        node = LayoutNode(
            id="btn",
            type=ComponentType.BUTTON,
            label="Click Me",
        )
        result = provider.transpile(node)
        assert "[Click Me]" in result

    @pytest.mark.unit
    def test_input_syntax(self, provider):
        """Input renders with quotes and underscores."""
        node = LayoutNode(
            id="field",
            type=ComponentType.INPUT,
            label="Username",
        )
        result = provider.transpile(node)
        assert '"Username' in result

    @pytest.mark.unit
    def test_text_syntax(self, provider):
        """Text renders with quotes."""
        node = LayoutNode(
            id="label",
            type=ComponentType.TEXT,
            label="Hello World",
        )
        result = provider.transpile(node)
        assert '"Hello World"' in result

    @pytest.mark.unit
    def test_horizontal_grid_layout(self, provider):
        """Horizontal orientation uses grid syntax."""
        node = LayoutNode(
            id="row",
            type=ComponentType.CONTAINER,
            orientation=Orientation.HORIZONTAL,
            children=[
                LayoutNode(id="a", type=ComponentType.BUTTON, label="A"),
                LayoutNode(id="b", type=ComponentType.BUTTON, label="B"),
            ],
        )
        result = provider.transpile(node)
        assert "{#" in result
        assert "[A] | [B]" in result

    @pytest.mark.unit
    def test_vertical_stacking(self, provider):
        """Vertical orientation stacks children."""
        node = LayoutNode(
            id="col",
            type=ComponentType.CONTAINER,
            orientation=Orientation.VERTICAL,
            children=[
                LayoutNode(id="a", type=ComponentType.BUTTON, label="A"),
                LayoutNode(id="b", type=ComponentType.BUTTON, label="B"),
            ],
        )
        result = provider.transpile(node)
        # Vertical should not have pipe separator
        assert "|" not in result or "{#" not in result

    @pytest.mark.unit
    def test_navbar_tabbed_style(self, provider):
        """Navbar uses tabbed panel style."""
        node = LayoutNode(
            id="nav",
            type=ComponentType.NAVBAR,
            children=[
                LayoutNode(id="home", type=ComponentType.BUTTON, label="Home"),
            ],
        )
        result = provider.transpile(node)
        assert "{+" in result

    @pytest.mark.unit
    def test_drawer_tabbed_style(self, provider):
        """Drawer uses tabbed panel style."""
        node = LayoutNode(
            id="side",
            type=ComponentType.DRAWER,
            children=[
                LayoutNode(id="menu", type=ComponentType.TEXT, label="Menu"),
            ],
        )
        result = provider.transpile(node)
        assert "{+" in result

    @pytest.mark.unit
    def test_nested_containers(self, provider):
        """Nested containers produce proper brace nesting."""
        layout = LayoutNode(
            id="outer",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(
                    id="inner",
                    type=ComponentType.CONTAINER,
                    children=[
                        LayoutNode(id="btn", type=ComponentType.BUTTON, label="Deep"),
                    ],
                ),
            ],
        )
        result = provider.transpile(layout)
        # Count braces - should be balanced
        open_braces = result.count("{")
        close_braces = result.count("}")
        assert open_braces == close_braces

    @pytest.mark.unit
    def test_label_in_container(self, provider):
        """Container label appears in output."""
        node = LayoutNode(
            id="panel",
            type=ComponentType.CONTAINER,
            label="Settings",
            children=[
                LayoutNode(id="opt", type=ComponentType.TEXT, label="Option"),
            ],
        )
        result = provider.transpile(node)
        assert '"Settings"' in result
