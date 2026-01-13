"""Unit tests for the Schema module."""

import pytest

from src.schema import (
    COMPONENT_REGISTRY,
    ComponentCategory,
    ComponentType,
    LayoutNodeSchema,
    Orientation,
    export_component_enum_schema,
    export_json_schema,
    export_llm_schema,
    get_component_category,
    get_component_meta,
    get_components_by_category,
    get_constraints,
    is_valid_layout_dict,
    resolve_alias,
    validate_component_type,
    validate_layout_dict,
    validate_orientation,
)


class TestComponentRegistry:
    """Tests for COMPONENT_REGISTRY completeness."""

    @pytest.mark.unit
    def test_all_component_types_registered(self):
        """Every ComponentType has metadata in registry."""
        for ct in ComponentType:
            assert ct in COMPONENT_REGISTRY, f"Missing metadata for {ct}"

    @pytest.mark.unit
    def test_registry_has_26_entries(self):
        """Registry contains exactly 26 component definitions."""
        assert len(COMPONENT_REGISTRY) == 26

    @pytest.mark.unit
    def test_all_entries_have_descriptions(self):
        """Every component has a non-empty description."""
        for ct, meta in COMPONENT_REGISTRY.items():
            assert meta.description, f"{ct} missing description"
            assert len(meta.description) > 10, f"{ct} description too short"

    @pytest.mark.unit
    def test_all_entries_have_category(self):
        """Every component is assigned a category."""
        for _ct, meta in COMPONENT_REGISTRY.items():
            assert isinstance(meta.category, ComponentCategory)


class TestComponentMeta:
    """Tests for ComponentMeta dataclass."""

    @pytest.mark.unit
    def test_meta_to_dict(self):
        """ComponentMeta converts to dictionary correctly."""
        meta = get_component_meta(ComponentType.BUTTON)
        d = meta.to_dict()

        assert d["type"] == "button"
        assert d["category"] == "control"
        assert "description" in d
        assert "aliases" in d
        assert isinstance(d["aliases"], list)

    @pytest.mark.unit
    def test_container_has_aliases(self):
        """Container type has expected aliases."""
        meta = get_component_meta(ComponentType.CONTAINER)
        assert "div" in meta.aliases
        assert "view" in meta.aliases

    @pytest.mark.unit
    def test_button_has_html_equivalents(self):
        """Button has HTML equivalent mappings."""
        meta = get_component_meta(ComponentType.BUTTON)
        assert "button" in meta.html_equivalents


class TestComponentConstraints:
    """Tests for component structural constraints."""

    @pytest.mark.unit
    def test_leaf_components_cannot_have_children(self):
        """Leaf components (text, icon, etc.) have children disabled."""
        leaf_types = [ComponentType.TEXT, ComponentType.ICON, ComponentType.IMAGE]
        for ct in leaf_types:
            constraints = get_constraints(ct)
            assert not constraints.can_have_children, f"{ct} should not allow children"

    @pytest.mark.unit
    def test_container_can_have_children(self):
        """Container types allow children."""
        constraints = get_constraints(ComponentType.CONTAINER)
        assert constraints.can_have_children

    @pytest.mark.unit
    def test_nav_has_max_children(self):
        """Navigation components have max children limit."""
        constraints = get_constraints(ComponentType.BOTTOM_NAV)
        assert constraints.max_children == 7


class TestCategoryLookup:
    """Tests for category lookup functions."""

    @pytest.mark.unit
    def test_get_component_category(self):
        """Category lookup returns correct values."""
        assert get_component_category(ComponentType.BUTTON) == ComponentCategory.CONTROL
        assert (
            get_component_category(ComponentType.CONTAINER)
            == ComponentCategory.CONTAINER
        )
        assert (
            get_component_category(ComponentType.TOOLBAR)
            == ComponentCategory.NAVIGATION
        )
        assert get_component_category(ComponentType.TEXT) == ComponentCategory.CONTENT

    @pytest.mark.unit
    def test_get_components_by_category(self):
        """Get all components in a category."""
        controls = get_components_by_category(ComponentCategory.CONTROL)
        assert ComponentType.BUTTON in controls
        assert ComponentType.INPUT in controls
        assert ComponentType.CONTAINER not in controls

    @pytest.mark.unit
    def test_all_categories_have_components(self):
        """Every category contains at least one component."""
        for cat in ComponentCategory:
            components = get_components_by_category(cat)
            assert len(components) > 0, f"Category {cat} has no components"


class TestAliasResolution:
    """Tests for alias resolution."""

    @pytest.mark.unit
    def test_resolve_direct_value(self):
        """Direct enum values resolve correctly."""
        assert resolve_alias("button") == ComponentType.BUTTON
        assert resolve_alias("container") == ComponentType.CONTAINER

    @pytest.mark.unit
    def test_resolve_alias(self):
        """Aliases resolve to canonical types."""
        assert resolve_alias("div") == ComponentType.CONTAINER
        assert resolve_alias("btn") == ComponentType.BUTTON
        assert resolve_alias("toggle") == ComponentType.SWITCH

    @pytest.mark.unit
    def test_resolve_case_insensitive(self):
        """Alias resolution is case-insensitive."""
        assert resolve_alias("DIV") == ComponentType.CONTAINER
        assert resolve_alias("Button") == ComponentType.BUTTON

    @pytest.mark.unit
    def test_resolve_unknown_returns_none(self):
        """Unknown aliases return None."""
        assert resolve_alias("unknown_component") is None
        assert resolve_alias("") is None


class TestSchemaGeneration:
    """Tests for schema generation functions."""

    @pytest.mark.unit
    def test_export_json_schema(self):
        """JSON schema exports with expected structure."""
        schema = export_json_schema()
        assert "$defs" in schema
        assert "LayoutNodeSchema" in schema["$defs"]
        node_schema = schema["$defs"]["LayoutNodeSchema"]
        assert "properties" in node_schema
        assert "id" in node_schema["properties"]
        assert "type" in node_schema["properties"]

    @pytest.mark.unit
    def test_export_component_enum_schema(self):
        """Component enum schema has all types with descriptions."""
        schema = export_component_enum_schema()
        assert len(schema) == 26
        assert "button" in schema
        assert "container" in schema
        assert isinstance(schema["button"], str)

    @pytest.mark.unit
    def test_export_llm_schema(self):
        """LLM schema contains all required sections."""
        schema = export_llm_schema()
        assert "schema" in schema
        assert "component_types" in schema
        assert "categories" in schema
        assert "orientation_options" in schema
        assert "constraints" in schema
        assert "examples" in schema

    @pytest.mark.unit
    def test_llm_schema_examples_are_valid(self):
        """LLM schema examples pass validation."""
        schema = export_llm_schema()
        for name, example in schema["examples"].items():
            errors = validate_layout_dict(example)
            assert errors == [], f"Example '{name}' has errors: {errors}"


class TestLayoutNodeSchema:
    """Tests for LayoutNodeSchema Pydantic model."""

    @pytest.mark.unit
    def test_create_minimal_node(self):
        """Create schema node with required fields."""
        node = LayoutNodeSchema(id="test", type=ComponentType.CONTAINER)
        assert node.id == "test"
        assert node.type == ComponentType.CONTAINER

    @pytest.mark.unit
    def test_defaults(self):
        """Default values are applied correctly."""
        node = LayoutNodeSchema(id="test", type=ComponentType.BUTTON)
        assert node.flex_ratio == 1
        assert node.orientation == Orientation.VERTICAL
        assert node.children == []
        assert node.label is None


class TestValidation:
    """Tests for schema validation functions."""

    @pytest.mark.unit
    def test_validate_component_type_valid(self):
        """Valid component types return ComponentType."""
        assert validate_component_type("button") == ComponentType.BUTTON
        assert validate_component_type("container") == ComponentType.CONTAINER

    @pytest.mark.unit
    def test_validate_component_type_alias(self):
        """Aliases resolve during validation."""
        assert validate_component_type("div") == ComponentType.CONTAINER

    @pytest.mark.unit
    def test_validate_component_type_invalid(self):
        """Invalid values return None."""
        assert validate_component_type("not_a_type") is None

    @pytest.mark.unit
    def test_validate_orientation_valid(self):
        """Valid orientations return Orientation."""
        assert validate_orientation("horizontal") == Orientation.HORIZONTAL
        assert validate_orientation("vertical") == Orientation.VERTICAL

    @pytest.mark.unit
    def test_validate_orientation_invalid(self):
        """Invalid orientations return None."""
        assert validate_orientation("diagonal") is None


class TestLayoutDictValidation:
    """Tests for validate_layout_dict function."""

    @pytest.mark.unit
    def test_valid_minimal_layout(self):
        """Minimal valid layout passes."""
        layout = {"id": "root", "type": "container"}
        errors = validate_layout_dict(layout)
        assert errors == []

    @pytest.mark.unit
    def test_valid_nested_layout(self):
        """Nested layout passes validation."""
        layout = {
            "id": "root",
            "type": "container",
            "children": [
                {"id": "child1", "type": "button", "label": "Click"},
                {"id": "child2", "type": "text", "label": "Hello"},
            ],
        }
        errors = validate_layout_dict(layout)
        assert errors == []

    @pytest.mark.unit
    def test_missing_id(self):
        """Missing id field is reported."""
        layout = {"type": "container"}
        errors = validate_layout_dict(layout)
        assert len(errors) == 1
        assert errors[0].error_type == "missing_field"
        assert "id" in errors[0].message

    @pytest.mark.unit
    def test_missing_type(self):
        """Missing type field is reported."""
        layout = {"id": "root"}
        errors = validate_layout_dict(layout)
        assert len(errors) == 1
        assert errors[0].error_type == "missing_field"

    @pytest.mark.unit
    def test_invalid_component_type(self):
        """Invalid component type is reported."""
        layout = {"id": "root", "type": "not_a_component"}
        errors = validate_layout_dict(layout)
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_enum"

    @pytest.mark.unit
    def test_invalid_flex_ratio_low(self):
        """Flex ratio below 1 is rejected."""
        layout = {"id": "root", "type": "container", "flex_ratio": 0}
        errors = validate_layout_dict(layout)
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_range"

    @pytest.mark.unit
    def test_invalid_flex_ratio_high(self):
        """Flex ratio above 12 is rejected."""
        layout = {"id": "root", "type": "container", "flex_ratio": 15}
        errors = validate_layout_dict(layout)
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_range"

    @pytest.mark.unit
    def test_invalid_orientation(self):
        """Invalid orientation is reported."""
        layout = {"id": "root", "type": "container", "orientation": "sideways"}
        errors = validate_layout_dict(layout)
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_enum"

    @pytest.mark.unit
    def test_duplicate_child_ids(self):
        """Duplicate IDs in children are detected."""
        layout = {
            "id": "root",
            "type": "container",
            "children": [
                {"id": "dupe", "type": "button"},
                {"id": "dupe", "type": "text"},
            ],
        }
        errors = validate_layout_dict(layout)
        assert any(e.error_type == "duplicate_id" for e in errors)

    @pytest.mark.unit
    def test_nested_errors_have_path(self):
        """Errors in nested nodes include path."""
        layout = {
            "id": "root",
            "type": "container",
            "children": [
                {"id": "child", "type": "invalid_type"},
            ],
        }
        errors = validate_layout_dict(layout)
        assert len(errors) == 1
        assert "children[0]" in errors[0].path


class TestIsValidLayoutDict:
    """Tests for is_valid_layout_dict convenience function."""

    @pytest.mark.unit
    def test_valid_returns_true(self):
        """Valid layout returns True."""
        layout = {"id": "root", "type": "container"}
        assert is_valid_layout_dict(layout) is True

    @pytest.mark.unit
    def test_invalid_returns_false(self):
        """Invalid layout returns False."""
        layout = {"id": "root", "type": "invalid"}
        assert is_valid_layout_dict(layout) is False
