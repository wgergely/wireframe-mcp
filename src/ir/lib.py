"""Intermediate Representation (IR) layer.

The IR layer serves as the **Transpilation Context**. It is the bridge
between the semantic MID models and the provider-specific DSL transpilers.

While currently proxying core MID models, this layer is intended for
implementation-specific transforms, optimizations, and context-aware
modifications required before a layout is converted to external formats
(e.g., D2, PlantUML).
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
