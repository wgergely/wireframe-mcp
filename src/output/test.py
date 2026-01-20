"""Tests for output module."""

import pytest

from src.ir import LayoutNode, TranspilationContext
from src.mid import ComponentType, Orientation
from src.output import LayoutOutput, OutputGenerator, format_layout_tree


@pytest.fixture
def dashboard_layout():
    """Create sample layout for testing."""
    return LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        label="Dashboard",
        orientation=Orientation.HORIZONTAL,
        children=[
            LayoutNode(
                id="sidebar",
                type=ComponentType.DRAWER,
                label="Sidebar",
                flex_ratio=3,
                children=[
                    LayoutNode(id="logo", type=ComponentType.IMAGE, label="Logo"),
                    LayoutNode(id="menu", type=ComponentType.LIST_ITEM, label="Menu"),
                ],
            ),
            LayoutNode(
                id="main",
                type=ComponentType.CONTAINER,
                label="Main Content",
                flex_ratio=9,
            ),
        ],
    )


class TestFormatLayoutTree:
    """Tests for format_layout_tree function."""

    @pytest.mark.unit
    def test_single_node(self):
        """Test formatting single node."""
        node = LayoutNode(id="root", type=ComponentType.CONTAINER, label="Root")
        result = format_layout_tree(node)
        assert "Root" in result
        assert "container" in result

    @pytest.mark.unit
    def test_nested_tree(self, dashboard_layout):
        """Test formatting nested tree."""
        result = format_layout_tree(dashboard_layout)
        assert "Dashboard" in result
        assert "Sidebar" in result
        assert "├──" in result or "└──" in result

    @pytest.mark.unit
    def test_horizontal_marker(self, dashboard_layout):
        """Test horizontal orientation is marked."""
        result = format_layout_tree(dashboard_layout)
        assert "horizontal" in result

    @pytest.mark.unit
    def test_flex_ratio_percentage(self, dashboard_layout):
        """Test flex ratio shown as percentage."""
        result = format_layout_tree(dashboard_layout)
        assert "25%" in result  # flex_ratio=3 -> 25%
        assert "75%" in result  # flex_ratio=9 -> 75%


class TestOutputGenerator:
    """Tests for OutputGenerator class."""

    @pytest.mark.unit
    def test_generate_d2(self, dashboard_layout):
        """Test generating D2 output."""
        gen = OutputGenerator(default_provider="d2")
        output = gen.generate(dashboard_layout)

        assert isinstance(output, LayoutOutput)
        assert output.provider == "d2"
        assert "Dashboard" in output.text_tree
        assert "Dashboard" in output.dsl_code

    @pytest.mark.unit
    def test_generate_plantuml(self, dashboard_layout):
        """Test generating PlantUML output."""
        gen = OutputGenerator()
        output = gen.generate(dashboard_layout, provider="plantuml")

        assert output.provider == "plantuml"
        assert "@startsalt" in output.dsl_code

    @pytest.mark.unit
    def test_output_contains_node(self, dashboard_layout):
        """Test output contains original node."""
        gen = OutputGenerator()
        output = gen.generate(dashboard_layout)
        assert output.node == dashboard_layout

    @pytest.mark.unit
    def test_generate_from_context(self, dashboard_layout):
        """Test generating output from TranspilationContext."""
        context = TranspilationContext(
            node=dashboard_layout,
            target_provider="plantuml",
        )
        gen = OutputGenerator()
        output = gen.generate_from_context(context)

        assert output.provider == "plantuml"
        assert output.node == dashboard_layout
        assert "@startsalt" in output.dsl_code
        assert "Dashboard" in output.text_tree

    @pytest.mark.unit
    def test_generate_from_context_uses_target_provider(self, dashboard_layout):
        """Context's target_provider overrides generator default."""
        context = TranspilationContext(
            node=dashboard_layout,
            target_provider="d2",
        )
        gen = OutputGenerator(default_provider="plantuml")
        output = gen.generate_from_context(context)

        # Should use context's provider, not generator's default
        assert output.provider == "d2"
        assert "{" in output.dsl_code  # D2 uses braces

    @pytest.mark.unit
    def test_generate_from_context_uses_context_default(self, dashboard_layout):
        """Context defaults to d2 when target_provider not specified."""
        context = TranspilationContext(node=dashboard_layout)
        gen = OutputGenerator(default_provider="plantuml")
        output = gen.generate_from_context(context)

        # TranspilationContext defaults target_provider to "d2"
        assert output.provider == "d2"

    @pytest.mark.unit
    def test_default_provider_d2(self, dashboard_layout):
        """Generator defaults to d2 provider."""
        gen = OutputGenerator()  # No explicit default
        output = gen.generate(dashboard_layout)
        assert output.provider == "d2"


class TestLayoutOutput:
    """Tests for LayoutOutput dataclass."""

    @pytest.mark.unit
    def test_layout_output_fields(self, dashboard_layout):
        """LayoutOutput has all expected fields."""
        output = LayoutOutput(
            text_tree="tree text",
            dsl_code="dsl code",
            node=dashboard_layout,
            provider="test",
        )
        assert output.text_tree == "tree text"
        assert output.dsl_code == "dsl code"
        assert output.node == dashboard_layout
        assert output.provider == "test"
        assert output.image_bytes is None
        assert output.image_format is None

    @pytest.mark.unit
    def test_layout_output_with_image(self, dashboard_layout):
        """LayoutOutput can include image data."""
        image_data = b"\x89PNG\r\n"
        output = LayoutOutput(
            text_tree="tree",
            dsl_code="dsl",
            node=dashboard_layout,
            provider="test",
            image_bytes=image_data,
            image_format="png",
        )
        assert output.image_bytes == image_data
        assert output.image_format == "png"


class TestFormatLayoutTreeEdgeCases:
    """Edge case tests for format_layout_tree."""

    @pytest.mark.unit
    def test_node_without_label_uses_id(self):
        """Node without label falls back to id."""
        node = LayoutNode(id="my-node-id", type=ComponentType.BUTTON)
        result = format_layout_tree(node)
        assert "my-node-id" in result

    @pytest.mark.unit
    def test_node_with_children_shows_tree_structure(self):
        """Multiple children show proper tree connectors."""
        node = LayoutNode(
            id="parent",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(id="child1", type=ComponentType.TEXT, label="First"),
                LayoutNode(id="child2", type=ComponentType.TEXT, label="Second"),
                LayoutNode(id="child3", type=ComponentType.TEXT, label="Third"),
            ],
        )
        result = format_layout_tree(node)
        # First two children should use ├──
        assert "├──" in result
        # Last child should use └──
        assert "└──" in result

    @pytest.mark.unit
    def test_deeply_nested_structure(self):
        """Deeply nested structure formats correctly."""
        child = LayoutNode(id="level3", type=ComponentType.ICON, label="Deep")
        mid = LayoutNode(
            id="level2",
            type=ComponentType.CARD,
            label="Middle",
            children=[child],
        )
        root = LayoutNode(
            id="level1",
            type=ComponentType.CONTAINER,
            label="Root",
            children=[mid],
        )
        result = format_layout_tree(root)

        lines = result.split("\n")
        assert len(lines) == 3
        assert "Root" in lines[0]
        assert "Middle" in lines[1]
        assert "Deep" in lines[2]

    @pytest.mark.unit
    def test_empty_children_list(self):
        """Node with empty children list formats as leaf."""
        node = LayoutNode(
            id="leaf",
            type=ComponentType.BUTTON,
            label="Leaf Node",
            children=[],
        )
        result = format_layout_tree(node)
        # Should just have one line (the node itself)
        assert result.count("\n") == 0
        assert "Leaf Node" in result

    @pytest.mark.unit
    def test_vertical_orientation_not_shown(self):
        """Vertical orientation (default) is not explicitly shown."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            orientation=Orientation.VERTICAL,
        )
        result = format_layout_tree(node)
        # Vertical is default, should not be in output
        assert "vertical" not in result.lower()

    @pytest.mark.unit
    def test_flex_ratio_one_not_shown(self):
        """Default flex_ratio=1 is not shown."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            flex_ratio=1,
        )
        result = format_layout_tree(node)
        # Default ratio should not show percentage
        assert "%" not in result


class TestOutputGeneratorProviderHandling:
    """Tests for OutputGenerator provider handling."""

    @pytest.mark.unit
    def test_invalid_provider_raises_error(self, dashboard_layout):
        """Invalid provider name raises KeyError."""
        gen = OutputGenerator()
        with pytest.raises(KeyError, match="Unknown provider"):
            gen.generate(dashboard_layout, provider="nonexistent_provider")
