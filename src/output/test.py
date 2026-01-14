"""Tests for output module."""

import pytest

from src.ir import LayoutNode
from src.mid import ComponentType, Orientation
from src.output import LayoutOutput, OutputGenerator, format_layout_tree


@pytest.fixture
def sample_layout():
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
    def test_nested_tree(self, sample_layout):
        """Test formatting nested tree."""
        result = format_layout_tree(sample_layout)
        assert "Dashboard" in result
        assert "Sidebar" in result
        assert "├──" in result or "└──" in result

    @pytest.mark.unit
    def test_horizontal_marker(self, sample_layout):
        """Test horizontal orientation is marked."""
        result = format_layout_tree(sample_layout)
        assert "horizontal" in result

    @pytest.mark.unit
    def test_flex_ratio_percentage(self, sample_layout):
        """Test flex ratio shown as percentage."""
        result = format_layout_tree(sample_layout)
        assert "25%" in result  # flex_ratio=3 -> 25%
        assert "75%" in result  # flex_ratio=9 -> 75%


class TestOutputGenerator:
    """Tests for OutputGenerator class."""

    @pytest.mark.unit
    def test_generate_d2(self, sample_layout):
        """Test generating D2 output."""
        gen = OutputGenerator(default_provider="d2")
        output = gen.generate(sample_layout)

        assert isinstance(output, LayoutOutput)
        assert output.provider == "d2"
        assert "Dashboard" in output.text_tree
        assert "Dashboard" in output.dsl_code

    @pytest.mark.unit
    def test_generate_plantuml(self, sample_layout):
        """Test generating PlantUML output."""
        gen = OutputGenerator()
        output = gen.generate(sample_layout, provider="plantuml")

        assert output.provider == "plantuml"
        assert "@startsalt" in output.dsl_code

    @pytest.mark.unit
    def test_output_contains_node(self, sample_layout):
        """Test output contains original node."""
        gen = OutputGenerator()
        output = gen.generate(sample_layout)
        assert output.node == sample_layout
