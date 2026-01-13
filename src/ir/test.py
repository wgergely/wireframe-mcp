"""Unit tests for IR models."""

import pytest

from src.ir import ComponentType, LayoutNode, Orientation, export_json_schema


class TestOrientation:
    """Tests for Orientation enum."""

    @pytest.mark.unit
    def test_enum_values(self):
        """All expected orientation values exist."""
        assert Orientation.HORIZONTAL.value == "horizontal"
        assert Orientation.VERTICAL.value == "vertical"
        assert Orientation.OVERLAY.value == "overlay"

    @pytest.mark.unit
    def test_string_serialization(self):
        """Enum serializes to string value."""
        assert str(Orientation.HORIZONTAL) == "Orientation.HORIZONTAL"
        assert Orientation.HORIZONTAL.value == "horizontal"


class TestComponentType:
    """Tests for ComponentType enum."""

    @pytest.mark.unit
    def test_all_types_exist(self):
        """All 24 Rico component types are defined."""
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
        }
        actual = {ct.value for ct in ComponentType}
        assert actual == expected
        assert (
            len(actual) == 26
        )  # 4 containers + 7 navigation + 5 content + 10 controls


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
        )
        assert node.id == "drawer"
        assert node.type == "drawer"
        assert node.label == "Navigation"
        assert node.flex_ratio == 3
        assert node.orientation == "horizontal"

    @pytest.mark.unit
    def test_nested_children(self):
        """Recursive children structure works correctly."""
        child = LayoutNode(id="child", type=ComponentType.BUTTON, label="Click")
        parent = LayoutNode(
            id="parent",
            type=ComponentType.CONTAINER,
            children=[child],
        )
        assert len(parent.children) == 1
        assert parent.children[0].id == "child"
        assert parent.children[0].label == "Click"

    @pytest.mark.unit
    def test_flex_ratio_minimum(self):
        """Flex ratio rejects values below 1."""
        with pytest.raises(ValueError):
            LayoutNode(id="test", type=ComponentType.CONTAINER, flex_ratio=0)

    @pytest.mark.unit
    def test_flex_ratio_maximum(self):
        """Flex ratio rejects values above 12."""
        with pytest.raises(ValueError):
            LayoutNode(id="test", type=ComponentType.CONTAINER, flex_ratio=13)

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

    @pytest.mark.unit
    def test_json_deserialization(self):
        """Node deserializes from JSON correctly."""
        data = {
            "id": "test",
            "type": "button",
            "label": "Submit",
        }
        node = LayoutNode.model_validate(data)
        assert node.id == "test"
        assert node.type == "button"
        assert node.label == "Submit"


class TestExportJsonSchema:
    """Tests for JSON schema export."""

    @pytest.mark.unit
    def test_schema_export(self):
        """Schema exports with expected structure."""
        schema = export_json_schema()
        # Pydantic v2 uses $defs with a $ref at root
        assert "$defs" in schema
        assert "LayoutNode" in schema["$defs"]
        assert schema["$defs"]["LayoutNode"]["title"] == "LayoutNode"

    @pytest.mark.unit
    def test_schema_has_required_fields(self):
        """Schema includes required fields."""
        schema = export_json_schema()
        layout_schema = schema["$defs"]["LayoutNode"]
        assert "id" in layout_schema["properties"]
        assert "type" in layout_schema["properties"]

    @pytest.mark.unit
    def test_schema_has_definitions(self):
        """Schema includes enum definitions."""
        schema = export_json_schema()
        assert "ComponentType" in schema["$defs"]
        assert "Orientation" in schema["$defs"]
