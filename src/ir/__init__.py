"""Intermediate Representation (IR) models for layout structures."""

from src.ir.lib import (
    COMPONENT_CATEGORIES,
    ComponentCategory,
    ComponentType,
    LayoutHint,
    LayoutNode,
    Orientation,
    TranspilationContext,
    export_json_schema,
    get_component_category,
)

__all__ = [
    # Core models
    "LayoutNode",
    "Orientation",
    "ComponentType",
    "ComponentCategory",
    "COMPONENT_CATEGORIES",
    "get_component_category",
    "export_json_schema",
    # Transpilation context
    "TranspilationContext",
    "LayoutHint",
]

