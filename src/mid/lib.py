"""Metadata-Intermediate-Definition (MID) layer.

The MID layer is the **Source of Truth** for the project's semantic layout notation.
It defines the structural model (LayoutNode) and validation rules that govern all
UI representations within the system.

This module acts as the "Contract" shared between the LLM generator,
the internal validation engine, and the transpilation layers.

Note: Component type definitions and schema generation are delegated to the
authoritative schema module (src/schema). This module re-exports those types
for backward compatibility.
"""

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field

# Import authoritative definitions from schema module
from src.schema import (
    COMPONENT_REGISTRY,
    AlignContent,
    Alignment,
    AlignSelf,
    ComponentCategory,
    ComponentType,
    Display,
    Justify,
    Orientation,
    SemanticColor,
    TextAlign,
    TextSize,
    TextTransform,
    TextWeight,
    Wrap,
    export_json_schema,
    get_component_category,
)


class LayoutNode(BaseModel):
    """Recursive node definition for the UI AST.

    Represents a single node in the layout tree, with support for
    nested children to form hierarchical UI structures.

    Attributes:
        id: Unique identifier for the node within the tree.
        type: Component type from the 26-category Rico taxonomy.
        label: Optional human-readable text content.
        flex_ratio: Grid span ratio (1-12) for relative sizing.
        children: Nested child nodes for hierarchical layouts.
        orientation: Layout flow direction for immediate children.
        align: Cross-axis alignment (align-items).
        justify: Main-axis distribution (justify-content).
        gap: Spacing between children in pixels.
        wrap: Overflow wrap behavior.
        padding: Internal padding in pixels.
        text_size: Text size hierarchy.
        text_weight: Font weight.
        text_transform: Text case transformation.
        text_align: Horizontal text alignment.
    """

    # Identity
    id: str = Field(..., description="Unique identifier for the node")
    type: ComponentType = Field(
        ..., description="Component type from restricted vocabulary"
    )

    # Content
    label: str | None = Field(None, description="Human-readable text content")

    # Structure
    children: list["LayoutNode"] = Field(
        default_factory=list,
        description="Nested child nodes for hierarchical layouts",
    )

    # Layout - existing
    flex_ratio: Annotated[int, Field(ge=1, le=12)] = Field(
        default=1,
        description="Grid span ratio (1-12 standard grid system)",
    )
    width: int | str | None = Field(
        default=None,
        description="Fixed width (px) or relative ('100%', 'auto')",
    )
    height: int | str | None = Field(
        default=None,
        description="Fixed height (px) or relative ('100%', 'auto')",
    )
    orientation: Orientation = Field(
        default=Orientation.VERTICAL,
        description="Layout flow direction for immediate children",
    )
    display: Display = Field(
        default=Display.FLEX,
        description="Layout mode: flex (default), grid, or block",
    )
    grid_columns: Annotated[int, Field(ge=1, le=12)] | None = Field(
        default=None,
        description="Number of grid columns (only when display=grid)",
    )
    grid_rows: Annotated[int, Field(ge=1)] | None = Field(
        default=None,
        description="Number of grid rows (only when display=grid)",
    )

    # Layout - new
    align: Alignment | None = Field(
        default=None,
        description="Cross-axis alignment (align-items)",
    )
    justify: Justify | None = Field(
        default=None,
        description="Main-axis distribution (justify-content)",
    )
    align_content: AlignContent | None = Field(
        default=None,
        description="Multi-line cross-axis alignment (align-content)",
    )
    gap: Annotated[int, Field(ge=0)] | None = Field(
        default=None,
        description="Spacing between children in pixels",
    )
    wrap: Wrap | None = Field(
        default=None,
        description="Overflow wrap behavior",
    )
    padding: Annotated[int, Field(ge=0)] | None = Field(
        default=None,
        description="Internal padding in pixels",
    )

    # Layout - Flex Item behavior
    align_self: AlignSelf | None = Field(
        default=None,
        description="Self alignment override (align-self)",
    )

    # Text styling
    text_size: TextSize | None = Field(
        default=None,
        description="Text size hierarchy (title/heading/body/caption)",
    )
    text_weight: TextWeight | None = Field(
        default=None,
        description="Font weight (light/normal/bold)",
    )
    text_transform: TextTransform | None = Field(
        default=None,
        description="Text case transformation",
    )
    text_align: TextAlign | None = Field(
        default=None,
        description="Horizontal text alignment",
    )

    # Visual Styling
    semantic_color: SemanticColor | None = Field(
        default=None,
        description="Semantic color status (e.g. primary, danger)",
    )

    # Interactive/Container props
    scrollable: bool = Field(
        default=False,
        description="Whether the container has overflow scrolling",
    )

    model_config = {
        "use_enum_values": True,
    }


# Re-export COMPONENT_CATEGORIES for backward compatibility
COMPONENT_CATEGORIES: dict[ComponentType, ComponentCategory] = {
    ct: COMPONENT_REGISTRY[ct].category for ct in ComponentType
}


@dataclass
class ValidationError:
    """Represents a validation error in a layout tree.

    Attributes:
        node_id: ID of the node where the error occurred.
        message: Human-readable error description.
        error_type: Machine-readable error classification.
    """

    node_id: str
    message: str
    error_type: str


def validate_layout(node: LayoutNode) -> list[ValidationError]:
    """Validate a LayoutNode tree for structural issues.

    Checks for:
    - Duplicate IDs within the tree
    - Cycles (nodes referencing ancestors)
    - flex_ratio values outside valid range 1-12

    Args:
        node: Root node of the layout tree to validate.

    Returns:
        List of ValidationError objects. Empty list if valid.
    """
    errors: list[ValidationError] = []

    # Track IDs, visited nodes, and path for cycle detection
    id_counts: dict[str, int] = {}
    visited: set[int] = set()

    # Import constraint getter
    from src.schema import get_constraints

    def visit(n: LayoutNode, path: set[int]) -> None:
        # Check for duplicate IDs
        id_counts[n.id] = id_counts.get(n.id, 0) + 1

        # Check for cycles
        obj_id = id(n)
        if obj_id in path:
            errors.append(
                ValidationError(
                    node_id=n.id,
                    message=f"Cycle detected: node '{n.id}' is its own ancestor",
                    error_type="cycle",
                )
            )
            return

        # Check flex ratio
        if n.flex_ratio < 1 or n.flex_ratio > 12:
            errors.append(
                ValidationError(
                    node_id=n.id,
                    message=f"flex_ratio {n.flex_ratio} outside valid range 1-12",
                    error_type="invalid_flex_ratio",
                )
            )

        # Check Component Constraints
        try:
            # Handle Pydantic use_enum_values=True (n.type might be str)
            comp_type = n.type
            if isinstance(comp_type, str):
                comp_type = ComponentType(comp_type)

            constraints = get_constraints(comp_type)

            # 1. Check can_have_children
            if not constraints.can_have_children and n.children:
                errors.append(
                    ValidationError(
                        node_id=n.id,
                        message=(
                            f"Component '{comp_type.value}' cannot have "
                            f"children (found {len(n.children)})"
                        ),
                        error_type="constraint_violation",
                    )
                )

            # 2. Check max_children
            if (
                constraints.max_children is not None
                and len(n.children) > constraints.max_children
            ):
                errors.append(
                    ValidationError(
                        node_id=n.id,
                        message=(
                            f"Component '{comp_type.value}' exceeds "
                            f"max_children {constraints.max_children} "
                            f"(found {len(n.children)})"
                        ),
                        error_type="constraint_violation",
                    )
                )

            # 3. Check allowed_child_types
            if constraints.allowed_child_types:
                for child in n.children:
                    # Convert child type to enum for comparison
                    child_type = child.type
                    if isinstance(child_type, str):
                        child_type = ComponentType(child_type)

                    if child_type not in constraints.allowed_child_types:
                        errors.append(
                            ValidationError(
                                node_id=n.id,
                                message=(
                                    f"Component '{comp_type.value}' cannot "
                                    f"contain child type '{child_type.value}'"
                                ),
                                error_type="constraint_violation",
                            )
                        )

            # 4. Check required_child_types (optional, not strictly enforced
            # by schema yet but good for future)
            # if constraints.required_child_types: ...

        except KeyError:
            # Component type not in registry (shouldn't happen with valid Enum)
            pass

        if obj_id in visited:
            return

        visited.add(obj_id)
        path.add(obj_id)
        for child in n.children:
            visit(child, path)
        path.remove(obj_id)

    visit(node, set())

    # Report duplicate IDs
    for node_id, count in id_counts.items():
        if count > 1:
            errors.append(
                ValidationError(
                    node_id=node_id,
                    message=f"Duplicate ID '{node_id}' appears {count} times",
                    error_type="duplicate_id",
                )
            )

    return errors


def is_valid(node: LayoutNode) -> bool:
    """Check if a layout tree is valid.

    Args:
        node: Root node of the layout tree.

    Returns:
        True if valid, False if any validation errors exist.
    """
    return not validate_layout(node)


__all__ = [
    # Re-exported from schema (for backward compatibility)
    "Orientation",
    "Display",
    "Alignment",
    "AlignSelf",
    "AlignContent",
    "Justify",
    "Wrap",
    "ComponentType",
    "ComponentCategory",
    "COMPONENT_CATEGORIES",
    "get_component_category",
    "export_json_schema",
    # MID-specific
    "LayoutNode",
    "ValidationError",
    "validate_layout",
    "is_valid",
]
