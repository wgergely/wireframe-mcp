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
    Alignment,
    ComponentCategory,
    ComponentType,
    Justify,
    LayoutNode,
    Orientation,
    TextAlign,
    TextSize,
    TextTransform,
    TextWeight,
    ValidationError,
    Wrap,
    export_json_schema,
    get_component_category,
    is_valid,
    validate_layout,
)

__all__ = [
    # Layout enums
    "Alignment",
    "Justify",
    "Orientation",
    "Wrap",
    # Text enums
    "TextAlign",
    "TextSize",
    "TextTransform",
    "TextWeight",
    # Component enums
    "ComponentType",
    "ComponentCategory",
    "COMPONENT_CATEGORIES",
    "get_component_category",
    # Core model
    "LayoutNode",
    "ValidationError",
    "validate_layout",
    "is_valid",
    "export_json_schema",
]
