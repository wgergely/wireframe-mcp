"""Schema module - authoritative source for UI layout schema definitions.

This module provides:
- Component type definitions with rich metadata
- JSON Schema generation for LLM integration
- Schema validation utilities
- Alias resolution for flexible component matching

Example usage:
    >>> from src.schema import export_llm_schema, validate_layout_dict
    >>> schema = export_llm_schema()  # For LLM prompt injection
    >>> errors = validate_layout_dict({"id": "root", "type": "container"})
"""

from .lib import (
    COMPONENT_REGISTRY,
    Alignment,
    ComponentCategory,
    ComponentConstraints,
    ComponentMeta,
    ComponentType,
    Justify,
    LayoutNodeSchema,
    Orientation,
    SchemaValidationError,
    TextAlign,
    TextSize,
    TextTransform,
    TextWeight,
    Wrap,
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

__all__ = [
    # Enums - Layout
    "Alignment",
    "ComponentCategory",
    "ComponentType",
    "Justify",
    "Orientation",
    "Wrap",
    # Enums - Text
    "TextAlign",
    "TextSize",
    "TextTransform",
    "TextWeight",
    # Metadata
    "ComponentConstraints",
    "ComponentMeta",
    "COMPONENT_REGISTRY",
    # Lookup functions
    "get_component_meta",
    "get_component_category",
    "get_components_by_category",
    "get_constraints",
    "resolve_alias",
    # Schema generation
    "LayoutNodeSchema",
    "export_json_schema",
    "export_component_enum_schema",
    "export_llm_schema",
    # Validation
    "SchemaValidationError",
    "validate_component_type",
    "validate_orientation",
    "validate_layout_dict",
    "is_valid_layout_dict",
]
