"""Intermediate Representation (IR) layer.

Convenience re-export of MID models. Reserved for future transpilation
context, optimizations, or provider-specific transforms.
"""

from src.mid import (
    COMPONENT_CATEGORIES,
    ComponentCategory,
    ComponentType,
    LayoutNode,
    Orientation,
    export_json_schema,
    get_component_category,
)

__all__ = [
    "Orientation",
    "ComponentType",
    "ComponentCategory",
    "COMPONENT_CATEGORIES",
    "get_component_category",
    "LayoutNode",
    "export_json_schema",
]
