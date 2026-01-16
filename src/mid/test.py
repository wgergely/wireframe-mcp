"""Unit tests for the MID layer."""

import pytest

from src.mid import (
    AlignContent,
    Alignment,
    AlignSelf,
    ComponentType,
    Justify,
    LayoutNode,
    Orientation,
    TextSize,
    TextWeight,
    Wrap,
    export_json_schema,
    validate_layout,
)


class TestLayoutEnums:
    """Tests for layout control enums."""

    @pytest.mark.unit
    def test_alignment_values(self):
        """All expected alignment values exist."""
        assert Alignment.START.value == "start"
        assert Alignment.CENTER.value == "center"
        assert Alignment.END.value == "end"
        assert Alignment.STRETCH.value == "stretch"

    @pytest.mark.unit
    def test_justify_values(self):
        """All expected justify values exist."""
        assert Justify.START.value == "start"
        assert Justify.CENTER.value == "center"
        assert Justify.END.value == "end"
        assert Justify.BETWEEN.value == "between"
        assert Justify.AROUND.value == "around"

    @pytest.mark.unit
    def test_wrap_values(self):
        """All expected wrap values exist."""
        assert Wrap.NONE.value == "none"
        assert Wrap.WRAP.value == "wrap"


class TestTextStylingEnums:
    """Tests for text styling enums."""

    @pytest.mark.unit
    def test_text_size_values(self):
        """All expected text size values exist."""
        assert TextSize.TITLE.value == "title"
        assert TextSize.HEADING.value == "heading"
        assert TextSize.BODY.value == "body"
        assert TextSize.CAPTION.value == "caption"

    @pytest.mark.unit
    def test_text_weight_values(self):
        """All expected text weight values exist."""
        assert TextWeight.LIGHT.value == "light"
        assert TextWeight.NORMAL.value == "normal"
        assert TextWeight.BOLD.value == "bold"


class TestOrientation:
    """Tests for Orientation enum."""

    @pytest.mark.unit
    def test_enum_values(self):
        """All expected orientation values exist."""
        assert Orientation.HORIZONTAL.value == "horizontal"
        assert Orientation.VERTICAL.value == "vertical"
        assert Orientation.OVERLAY.value == "overlay"


class TestComponentType:
    """Tests for ComponentType enum."""

    @pytest.mark.unit
    def test_all_types_exist(self):
        """All expected component types are defined."""
        expected = {
            # Containers
            "container",
            "card",
            "modal",
            "web_view",
            # Navigation
            "toolbar",
            "navbar",
            "bottom_nav",
            "drawer",
            "tab_bar",
            "multi_tab",
            "pager_indicator",
            # Content
            "text",
            "image",
            "list_item",
            "icon",
            "advertisement",
            # Controls
            "button",
            "text_button",
            "input",
            "checkbox",
            "radio_button",
            "switch",
            "slider",
            "spinner",
            "date_picker",
            "number_stepper",
            "tree",
            "menu_bar",
            "data_grid",
            "divider",
        }
        actual = {ct.value for ct in ComponentType}
        assert actual == expected
        assert len(actual) == 30


class TestLayoutNode:
    """Tests for LayoutNode model."""

    @pytest.mark.unit
    def test_minimal_node(self):
        """Create node with only required fields."""
        node = LayoutNode(id="test", type=ComponentType.CONTAINER)
        assert node.id == "test"
        assert node.type == "container"
        assert node.label is None
        assert node.flex_ratio == 1
        assert node.children == []
        assert node.orientation == "vertical"

    @pytest.mark.unit
    def test_full_node(self):
        """Create node with all fields specified."""
        node = LayoutNode(
            id="drawer",
            type=ComponentType.DRAWER,
            label="Navigation",
            flex_ratio=3,
            orientation=Orientation.HORIZONTAL,
            align_self=AlignSelf.STRETCH,
            align_content=AlignContent.BETWEEN,
        )
        assert node.id == "drawer"
        assert node.type == "drawer"
        assert node.label == "Navigation"
        assert node.flex_ratio == 3
        assert node.orientation == "horizontal"
        assert node.align_self == "stretch"
        assert node.width is None
        assert node.height is None

    @pytest.mark.unit
    def test_fixed_sizing(self):
        """Node accepts fixed sizing."""
        node = LayoutNode(
            id="fixed",
            type=ComponentType.CONTAINER,
            width=200,
            height="50%",
        )
        assert node.width == 200
        assert node.height == "50%"

    @pytest.mark.unit
    def test_flex_ratio_valid_range(self):
        """Flex ratio accepts values 1-12."""
        for ratio in range(1, 13):
            node = LayoutNode(id="test", type=ComponentType.CONTAINER, flex_ratio=ratio)
            assert node.flex_ratio == ratio

    @pytest.mark.unit
    def test_json_serialization(self):
        """Node serializes to JSON correctly."""
        node = LayoutNode(
            id="test",
            type=ComponentType.BUTTON,
            label="Submit",
        )
        data = node.model_dump()
        assert data["id"] == "test"
        assert data["type"] == "button"
        assert data["label"] == "Submit"


class TestValidateLayout:
    """Tests for validate_layout function."""

    @pytest.mark.unit
    def test_valid_tree(self):
        """Well-formed tree passes validation."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(id="child1", type=ComponentType.BUTTON),
                LayoutNode(id="child2", type=ComponentType.TEXT),
            ],
        )
        errors = validate_layout(node)
        assert errors == []

    @pytest.mark.unit
    def test_duplicate_ids(self):
        """Duplicate IDs are detected."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(id="dupe", type=ComponentType.BUTTON),
                LayoutNode(id="dupe", type=ComponentType.TEXT),
            ],
        )
        errors = validate_layout(node)
        assert len(errors) == 1
        assert errors[0].error_type == "duplicate_id"
        assert "dupe" in errors[0].message


class TestExportJsonSchema:
    """Tests for JSON schema export."""

    @pytest.mark.unit
    def test_schema_export(self):
        """Schema exports with expected structure."""
        schema = export_json_schema()
        assert "$defs" in schema
        assert "LayoutNodeSchema" in schema["$defs"]
        assert schema["$defs"]["LayoutNodeSchema"]["title"] == "LayoutNodeSchema"
