"""Unit tests for validation module."""

import pytest

from src.ir import ComponentType, LayoutNode
from src.validation import ValidationError, is_valid, validate_layout


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

    @pytest.mark.unit
    def test_multiple_duplicate_ids(self):
        """Multiple duplicate ID groups are detected."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(id="dupe_a", type=ComponentType.BUTTON),
                LayoutNode(id="dupe_a", type=ComponentType.TEXT),
                LayoutNode(id="dupe_b", type=ComponentType.INPUT),
                LayoutNode(id="dupe_b", type=ComponentType.IMAGE),
            ],
        )
        errors = validate_layout(node)
        assert len(errors) == 2
        error_ids = {e.node_id for e in errors}
        assert error_ids == {"dupe_a", "dupe_b"}

    @pytest.mark.unit
    def test_deeply_nested_duplicates(self):
        """Duplicates in deeply nested trees are detected."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(
                    id="level1",
                    type=ComponentType.CONTAINER,
                    children=[
                        LayoutNode(
                            id="level2",
                            type=ComponentType.CONTAINER,
                            children=[
                                LayoutNode(id="root", type=ComponentType.BUTTON),
                            ],
                        ),
                    ],
                ),
            ],
        )
        errors = validate_layout(node)
        assert len(errors) == 1
        assert errors[0].node_id == "root"

    @pytest.mark.unit
    def test_single_node_valid(self):
        """Single node without children is valid."""
        node = LayoutNode(id="solo", type=ComponentType.BUTTON)
        errors = validate_layout(node)
        assert errors == []


class TestIsValid:
    """Tests for is_valid convenience function."""

    @pytest.mark.unit
    def test_valid_returns_true(self):
        """Valid tree returns True."""
        node = LayoutNode(id="root", type=ComponentType.CONTAINER)
        assert is_valid(node) is True

    @pytest.mark.unit
    def test_invalid_returns_false(self):
        """Invalid tree returns False."""
        node = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(id="root", type=ComponentType.BUTTON),
            ],
        )
        assert is_valid(node) is False


class TestValidationError:
    """Tests for ValidationError dataclass."""

    @pytest.mark.unit
    def test_error_attributes(self):
        """ValidationError has expected attributes."""
        error = ValidationError(
            node_id="test",
            message="Test error",
            error_type="test_type",
        )
        assert error.node_id == "test"
        assert error.message == "Test error"
        assert error.error_type == "test_type"
