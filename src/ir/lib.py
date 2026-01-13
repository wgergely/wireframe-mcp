"""Core IR models for layout representation.

This module defines the Intermediate Representation (IR) schema that serves
as the contract between LLM output and the transpilation layer. The LLM
produces structured JSON conforming to these models, which is then validated
and transpiled to provider-specific DSL syntax.
"""

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
    providing a standardized vocabulary for UI components that enables
    high-fidelity LLM mapping and multi-target transpilation.

    Categories:
        Containers: container, card, modal, web_view
        Navigation: toolbar, navbar, bottom_nav, drawer, tab_bar,
                    multi_tab, pager_indicator
        Content: text, image, list_item, icon, advertisement
        Controls: button, text_button, input, checkbox, radio_button,
                  switch, slider, spinner, date_picker, number_stepper
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
    """Get the category for a component type.

    Args:
        component_type: The component type to categorize.

    Returns:
        ComponentCategory: The category this component belongs to.
    """
    return COMPONENT_CATEGORIES[component_type]


class LayoutNode(BaseModel):
    """Recursive node definition for the UI AST.

    Each node represents a UI component with optional children,
    forming a tree structure that describes the layout hierarchy.

    Attributes:
        id: Unique identifier for the node. Used for validation and reference.
        type: The component type from the restricted vocabulary.
        label: Optional human-readable text content or label.
        flex_ratio: Grid span ratio (1-12 standard grid system).
        children: Nested child nodes for hierarchical layouts.
        orientation: Layout flow direction for immediate children.

    Example:
        >>> node = LayoutNode(
        ...     id="drawer",
        ...     type=ComponentType.DRAWER,
        ...     label="Navigation",
        ...     flex_ratio=3,
        ... )
    """

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


def export_json_schema() -> dict:
    """Export the LayoutNode JSON Schema for LLM prompt injection.

    Returns a JSON Schema dictionary that can be provided to LLMs
    to constrain their output to valid LayoutNode structures.

    Returns:
        dict: JSON Schema representation of LayoutNode.

    Example:
        >>> schema = export_json_schema()
        >>> schema["title"]
        'LayoutNode'
    """
    return LayoutNode.model_json_schema()
