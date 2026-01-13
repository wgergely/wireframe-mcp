"""Metadata-Intermediate-Definition (MID) layer.

The MID layer is the **Source of Truth** for the project's semantic layout notation.
It defines the core vocabulary (ComponentType), structural model (LayoutNode),
and validation rules that govern all UI representations within the system.

This module acts as the "Contract" shared between the LLM generator,
the internal validation engine, and the transpilation layers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class Orientation(str, Enum):
    """Defines the flow direction of a container.

    Maps to CSS flex-direction concepts:
    - HORIZONTAL: Children flow left-to-right (flex-row)
    - VERTICAL: Children flow top-to-bottom (flex-col)
    - OVERLAY: Children stack on z-axis (absolute positioning)
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    OVERLAY = "overlay"


class ComponentType(str, Enum):
    """Rico-based UI component taxonomy (26 categories).

    These categories are derived from Rico dataset semantic annotations,
    providing a standardized vocabulary for UI components.
    """

    # Containers
    CONTAINER = "container"
    CARD = "card"
    MODAL = "modal"
    WEB_VIEW = "web_view"

    # Navigation
    TOOLBAR = "toolbar"
    NAVBAR = "navbar"
    BOTTOM_NAV = "bottom_nav"
    DRAWER = "drawer"
    TAB_BAR = "tab_bar"
    MULTI_TAB = "multi_tab"
    PAGER_INDICATOR = "pager_indicator"

    # Content
    TEXT = "text"
    IMAGE = "image"
    LIST_ITEM = "list_item"
    ICON = "icon"
    ADVERTISEMENT = "advertisement"

    # Controls
    BUTTON = "button"
    TEXT_BUTTON = "text_button"
    INPUT = "input"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    SWITCH = "switch"
    SLIDER = "slider"
    SPINNER = "spinner"
    DATE_PICKER = "date_picker"
    NUMBER_STEPPER = "number_stepper"


class ComponentCategory(str, Enum):
    """High-level component groupings for filtering and analysis."""

    CONTAINER = "container"
    NAVIGATION = "navigation"
    CONTENT = "content"
    CONTROL = "control"


# Mapping from ComponentType to its category
COMPONENT_CATEGORIES: dict[ComponentType, ComponentCategory] = {
    # Containers
    ComponentType.CONTAINER: ComponentCategory.CONTAINER,
    ComponentType.CARD: ComponentCategory.CONTAINER,
    ComponentType.MODAL: ComponentCategory.CONTAINER,
    ComponentType.WEB_VIEW: ComponentCategory.CONTAINER,
    # Navigation
    ComponentType.TOOLBAR: ComponentCategory.NAVIGATION,
    ComponentType.NAVBAR: ComponentCategory.NAVIGATION,
    ComponentType.BOTTOM_NAV: ComponentCategory.NAVIGATION,
    ComponentType.DRAWER: ComponentCategory.NAVIGATION,
    ComponentType.TAB_BAR: ComponentCategory.NAVIGATION,
    ComponentType.MULTI_TAB: ComponentCategory.NAVIGATION,
    ComponentType.PAGER_INDICATOR: ComponentCategory.NAVIGATION,
    # Content
    ComponentType.TEXT: ComponentCategory.CONTENT,
    ComponentType.IMAGE: ComponentCategory.CONTENT,
    ComponentType.LIST_ITEM: ComponentCategory.CONTENT,
    ComponentType.ICON: ComponentCategory.CONTENT,
    ComponentType.ADVERTISEMENT: ComponentCategory.CONTENT,
    # Controls
    ComponentType.BUTTON: ComponentCategory.CONTROL,
    ComponentType.TEXT_BUTTON: ComponentCategory.CONTROL,
    ComponentType.INPUT: ComponentCategory.CONTROL,
    ComponentType.CHECKBOX: ComponentCategory.CONTROL,
    ComponentType.RADIO_BUTTON: ComponentCategory.CONTROL,
    ComponentType.SWITCH: ComponentCategory.CONTROL,
    ComponentType.SLIDER: ComponentCategory.CONTROL,
    ComponentType.SPINNER: ComponentCategory.CONTROL,
    ComponentType.DATE_PICKER: ComponentCategory.CONTROL,
    ComponentType.NUMBER_STEPPER: ComponentCategory.CONTROL,
}


def get_component_category(component_type: ComponentType) -> ComponentCategory:
    """Get the category for a component type."""
    return COMPONENT_CATEGORIES[component_type]


class LayoutNode(BaseModel):
    """Recursive node definition for the UI AST."""

    id: str = Field(..., description="Unique identifier for the node")
    type: ComponentType = Field(
        ..., description="Component type from restricted vocabulary"
    )
    label: str | None = Field(None, description="Human-readable text content")
    flex_ratio: Annotated[int, Field(ge=1, le=12)] = Field(
        default=1,
        description="Grid span ratio (1-12 standard grid system)",
    )
    children: list["LayoutNode"] = Field(
        default_factory=list,
        description="Nested child nodes for hierarchical layouts",
    )
    orientation: Orientation = Field(
        default=Orientation.VERTICAL,
        description="Layout flow direction for immediate children",
    )

    model_config = {
        "use_enum_values": True,
    }


@dataclass
class ValidationError:
    """Represents a validation error in a layout tree."""

    node_id: str
    message: str
    error_type: str


def validate_layout(node: LayoutNode) -> list[ValidationError]:
    """Validate a LayoutNode tree for structural issues."""
    errors: list[ValidationError] = []

    # Track IDs, visited nodes, and path for cycle detection
    id_counts: dict[str, int] = {}
    visited: set[int] = set()

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
    """Check if a layout tree is valid."""
    return not validate_layout(node)


def export_json_schema() -> dict:
    """Export the LayoutNode JSON Schema for LLM prompt injection."""
    return LayoutNode.model_json_schema()


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
]  # Removed: _collect_ids, _validate_flex_ratios, _detect_cycles (consolidated into validate_layout)
