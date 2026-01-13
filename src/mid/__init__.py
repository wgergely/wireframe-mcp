"""MID layer - Metadata-Intermediate-Definition for UI layouts.

This module provides the semantic layout model (LayoutNode) and validation.
Component type definitions are delegated to the authoritative schema module.

Example usage:
    >>> from src.mid import LayoutNode, ComponentType, validate_layout
    >>> node = LayoutNode(id="root", type=ComponentType.CONTAINER)
    >>> errors = validate_layout(node)
"""

from .lib import (
    COMPONENT_CATEGORIES,
    ComponentCategory,
    ComponentType,
    LayoutNode,
    Orientation,
    ValidationError,
    export_json_schema,
    get_component_category,
    is_valid,
    validate_layout,
)

__all__ = [
    "Orientation",
    "ComponentType",
    "ComponentCategory",
    "COMPONENT_CATEGORIES",
    "get_component_category",
    "LayoutNode",
    "ValidationError",
    "validate_layout",
    "is_valid",
    "export_json_schema",
]
