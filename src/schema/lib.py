"""Authoritative Schema Module for UI Layout Definitions.

This module serves as the single source of truth for all schema knowledge in
the wireframe system. It provides:
- Rich component metadata (descriptions, constraints, aliases)
- Programmatic schema generation based on enums
- Schema verification and validation
- LLM-optimized schema exports

All schema-related queries should route through this module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ComponentCategory(str, Enum):
    """High-level component groupings."""

    CONTAINER = "container"
    NAVIGATION = "navigation"
    CONTENT = "content"
    CONTROL = "control"


class Orientation(str, Enum):
    """Layout flow direction for container children.

    Maps to CSS flex-direction concepts:
    - HORIZONTAL: Children flow left-to-right (flex-row)
    - VERTICAL: Children flow top-to-bottom (flex-col)
    - OVERLAY: Children stack on z-axis (absolute positioning)
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    OVERLAY = "overlay"


class Alignment(str, Enum):
    """Cross-axis alignment for container children (CSS align-items).

    Controls how children align perpendicular to the main axis:
    - START: Align to start edge
    - CENTER: Center along cross-axis
    - END: Align to end edge
    - STRETCH: Stretch to fill container (default)
    """

    START = "start"
    CENTER = "center"
    END = "end"
    STRETCH = "stretch"


class Justify(str, Enum):
    """Main-axis distribution for container children (CSS justify-content).

    Controls how space is distributed along the main axis:
    - START: Pack children at start
    - CENTER: Center children
    - END: Pack children at end
    - BETWEEN: Distribute with space between
    - AROUND: Distribute with space around
    """

    START = "start"
    CENTER = "center"
    END = "end"
    BETWEEN = "between"
    AROUND = "around"
    EVENLY = "evenly"


class AlignContent(str, Enum):
    """Multi-line cross-axis alignment (CSS align-content).

    Controls how lines align in a multi-line flex container:
    - START: Pack lines at start
    - CENTER: Pack lines at center
    - END: Pack lines at end
    - BETWEEN: Distribute lines with space between
    - AROUND: Distribute lines with space around
    - STRETCH: Stretch lines to fill container
    """

    START = "start"
    CENTER = "center"
    END = "end"
    BETWEEN = "between"
    AROUND = "around"
    STRETCH = "stretch"


class AlignSelf(str, Enum):
    """Item-specific cross-axis alignment (CSS align-self).

    Overrides the container's align-items value for a specific child:
    - AUTO: Inherit from container
    - START: Align to start edge
    - CENTER: Center along cross-axis
    - END: Align to end edge
    - STRETCH: Stretch to fill container
    """

    AUTO = "auto"
    START = "start"
    CENTER = "center"
    END = "end"
    STRETCH = "stretch"


class Wrap(str, Enum):
    """Wrap behavior for container overflow (CSS flex-wrap)."""

    NONE = "none"
    WRAP = "wrap"


class Display(str, Enum):
    """Display/layout mode for containers.

    Determines how children are laid out:
    - FLEX: Flexbox layout (default) - single-axis distribution
    - GRID: CSS Grid layout - explicit rows/columns with cell placement
    - BLOCK: Block layout - simple vertical stacking without flex features
    """

    FLEX = "flex"
    GRID = "grid"
    BLOCK = "block"


class TextSize(str, Enum):
    """Text size hierarchy for visual importance.

    Maps to typical heading/body scales:
    - TITLE: Large display text (h1-h2 equivalent)
    - HEADING: Section headings (h3-h4 equivalent)
    - BODY: Default body text
    - CAPTION: Small supporting text
    """

    TITLE = "title"
    HEADING = "heading"
    BODY = "body"
    CAPTION = "caption"


class TextWeight(str, Enum):
    """Font weight for text emphasis."""

    LIGHT = "light"
    NORMAL = "normal"
    BOLD = "bold"


class TextTransform(str, Enum):
    """Text case transformation (CSS text-transform)."""

    NONE = "none"
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    CAPITALIZE = "capitalize"


class TextAlign(str, Enum):
    """Horizontal text alignment."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class SemanticColor(str, Enum):
    """Semantic color palette for wireframe elements."""

    DEFAULT = "default"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"
    INFO = "info"


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
    MENU_BAR = "menu_bar"
    TREE = "tree"

    # Content
    TEXT = "text"
    IMAGE = "image"

    LIST_ITEM = "list_item"
    DATA_GRID = "data_grid"
    DIVIDER = "divider"
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


@dataclass(frozen=True)
class ComponentConstraints:
    """Structural constraints for a component type."""

    can_have_children: bool = True
    max_children: int | None = None
    allowed_child_types: tuple[ComponentType, ...] | None = None
    required_child_types: tuple[ComponentType, ...] | None = None
    default_orientation: Orientation = Orientation.VERTICAL
    supports_label: bool = True
    min_flex_ratio: int = 1
    max_flex_ratio: int = 12


@dataclass(frozen=True)
class ComponentMeta:
    """Rich metadata definition for a UI component type.

    Provides comprehensive information about each component including
    descriptions, categorization, aliases, and structural constraints.
    """

    type: ComponentType
    category: ComponentCategory
    description: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    html_equivalents: tuple[str, ...] = field(default_factory=tuple)
    android_classes: tuple[str, ...] = field(default_factory=tuple)
    constraints: ComponentConstraints = field(default_factory=ComponentConstraints)
    examples: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for schema export."""
        return {
            "type": self.type.value,
            "category": self.category.value,
            "description": self.description,
            "aliases": list(self.aliases),
            "html_equivalents": list(self.html_equivalents),
            "android_classes": list(self.android_classes),
            "constraints": {
                "can_have_children": self.constraints.can_have_children,
                "max_children": self.constraints.max_children,
                "default_orientation": self.constraints.default_orientation.value,
                "supports_label": self.constraints.supports_label,
            },
            "examples": list(self.examples),
        }


# Leaf node constraints (no children)
_LEAF_CONSTRAINTS = ComponentConstraints(can_have_children=False, max_children=0)

# Control constraints (typically no children)
_CONTROL_CONSTRAINTS = ComponentConstraints(
    can_have_children=False,
    max_children=0,
    supports_label=True,
)

# Container constraints (horizontal by default)
_HORIZONTAL_CONTAINER = ComponentConstraints(
    default_orientation=Orientation.HORIZONTAL,
)

_VERTICAL_CONTAINER = ComponentConstraints(
    default_orientation=Orientation.VERTICAL,
)

# Navigation bar constraints
_NAV_CONSTRAINTS = ComponentConstraints(
    default_orientation=Orientation.HORIZONTAL,
    max_children=7,
)


COMPONENT_REGISTRY: dict[ComponentType, ComponentMeta] = {
    # === CONTAINERS ===
    ComponentType.CONTAINER: ComponentMeta(
        type=ComponentType.CONTAINER,
        category=ComponentCategory.CONTAINER,
        description="Generic layout container for grouping child elements",
        aliases=("div", "view", "frame", "box", "wrapper", "section"),
        html_equivalents=("div", "section", "article", "main"),
        android_classes=(
            "android.view.ViewGroup",
            "android.widget.FrameLayout",
            "android.widget.LinearLayout",
        ),
        constraints=ComponentConstraints(),
        examples=("Page wrapper", "Content section", "Form container"),
    ),
    ComponentType.CARD: ComponentMeta(
        type=ComponentType.CARD,
        category=ComponentCategory.CONTAINER,
        description="Elevated content container with visual separation",
        aliases=("panel", "tile", "block", "surface"),
        html_equivalents=("div.card", "article"),
        android_classes=("androidx.cardview.widget.CardView",),
        constraints=ComponentConstraints(),
        examples=("Product card", "User profile card", "Dashboard widget"),
    ),
    ComponentType.MODAL: ComponentMeta(
        type=ComponentType.MODAL,
        category=ComponentCategory.CONTAINER,
        description="Overlay dialog that interrupts workflow for focused interaction",
        aliases=("dialog", "popup", "overlay", "lightbox", "sheet"),
        html_equivalents=("dialog", "div.modal"),
        android_classes=("android.app.Dialog", "android.app.AlertDialog"),
        constraints=ComponentConstraints(default_orientation=Orientation.OVERLAY),
        examples=("Confirmation dialog", "Settings modal", "Image lightbox"),
    ),
    ComponentType.WEB_VIEW: ComponentMeta(
        type=ComponentType.WEB_VIEW,
        category=ComponentCategory.CONTAINER,
        description="Embedded web content renderer",
        aliases=("webview", "iframe", "browser", "embedded"),
        html_equivalents=("iframe",),
        android_classes=("android.webkit.WebView",),
        constraints=ComponentConstraints(can_have_children=False),
        examples=("Embedded documentation", "Third-party widget", "HTML content"),
    ),
    # === NAVIGATION ===
    ComponentType.TOOLBAR: ComponentMeta(
        type=ComponentType.TOOLBAR,
        category=ComponentCategory.NAVIGATION,
        description="Top action bar with title and action buttons",
        aliases=("appbar", "header", "actionbar", "topbar"),
        html_equivalents=("header", "nav.toolbar"),
        android_classes=(
            "android.widget.Toolbar",
            "androidx.appcompat.widget.Toolbar",
        ),
        constraints=_HORIZONTAL_CONTAINER,
        examples=("App header with back button", "Page title bar"),
    ),
    ComponentType.NAVBAR: ComponentMeta(
        type=ComponentType.NAVBAR,
        category=ComponentCategory.NAVIGATION,
        description="Primary navigation menu bar",
        aliases=("navigation", "menubar", "nav"),
        html_equivalents=("nav",),
        android_classes=(),
        constraints=_NAV_CONSTRAINTS,
        examples=("Site navigation", "Main menu"),
    ),
    ComponentType.BOTTOM_NAV: ComponentMeta(
        type=ComponentType.BOTTOM_NAV,
        category=ComponentCategory.NAVIGATION,
        description="Fixed bottom navigation bar for primary destinations",
        aliases=("bottombar", "tab_navigation", "footer_nav"),
        html_equivalents=("nav.bottom",),
        android_classes=(
            "com.google.android.material.bottomnavigation.BottomNavigationView",
        ),
        constraints=_NAV_CONSTRAINTS,
        examples=("Home/Search/Profile tabs", "App main sections"),
    ),
    ComponentType.DRAWER: ComponentMeta(
        type=ComponentType.DRAWER,
        category=ComponentCategory.NAVIGATION,
        description="Slide-out navigation panel from screen edge",
        aliases=("sidebar", "sidenav", "navigation_drawer", "menu_drawer"),
        html_equivalents=("aside", "nav.drawer"),
        android_classes=("androidx.drawerlayout.widget.DrawerLayout",),
        constraints=ComponentConstraints(),
        examples=("Hamburger menu content", "Settings drawer"),
    ),
    ComponentType.TAB_BAR: ComponentMeta(
        type=ComponentType.TAB_BAR,
        category=ComponentCategory.NAVIGATION,
        description="Horizontal tab selector for content switching",
        aliases=("tabs", "tabstrip", "segmented_control"),
        html_equivalents=("nav.tabs", "ul.tabs"),
        android_classes=("com.google.android.material.tabs.TabLayout",),
        constraints=_NAV_CONSTRAINTS,
        examples=("Settings sections", "Content categories"),
    ),
    ComponentType.MULTI_TAB: ComponentMeta(
        type=ComponentType.MULTI_TAB,
        category=ComponentCategory.NAVIGATION,
        description="Scrollable tab container for many options",
        aliases=("scrollable_tabs", "tab_scroller"),
        html_equivalents=("nav.tabs.scrollable",),
        android_classes=("com.google.android.material.tabs.TabLayout",),
        constraints=_HORIZONTAL_CONTAINER,
        examples=("Category browser", "Filter options"),
    ),
    ComponentType.PAGER_INDICATOR: ComponentMeta(
        type=ComponentType.PAGER_INDICATOR,
        category=ComponentCategory.NAVIGATION,
        description="Dot indicators showing current page in carousel/pager",
        aliases=("dots", "page_dots", "carousel_indicator", "stepper"),
        html_equivalents=("div.pagination-dots",),
        android_classes=(),
        constraints=_LEAF_CONSTRAINTS,
        examples=("Onboarding progress", "Image carousel position"),
    ),
    # === CONTENT ===
    ComponentType.TEXT: ComponentMeta(
        type=ComponentType.TEXT,
        category=ComponentCategory.CONTENT,
        description="Static text display element",
        aliases=("label", "paragraph", "heading", "title", "caption", "span"),
        html_equivalents=("p", "span", "h1", "h2", "h3", "h4", "h5", "h6", "label"),
        android_classes=("android.widget.TextView",),
        constraints=_LEAF_CONSTRAINTS,
        examples=("Page title", "Body text", "Form label"),
    ),
    ComponentType.IMAGE: ComponentMeta(
        type=ComponentType.IMAGE,
        category=ComponentCategory.CONTENT,
        description="Visual image or graphic display",
        aliases=("img", "picture", "photo", "graphic", "illustration"),
        html_equivalents=("img", "picture", "svg"),
        android_classes=("android.widget.ImageView",),
        constraints=_LEAF_CONSTRAINTS,
        examples=("Product photo", "Hero image", "Avatar"),
    ),
    ComponentType.LIST_ITEM: ComponentMeta(
        type=ComponentType.LIST_ITEM,
        category=ComponentCategory.CONTENT,
        description="Single item in a list or collection",
        aliases=("item", "row", "entry", "list_row"),
        html_equivalents=("li", "div.list-item"),
        android_classes=(),
        constraints=_HORIZONTAL_CONTAINER,
        examples=("Contact entry", "Message preview", "Settings option"),
    ),
    ComponentType.ICON: ComponentMeta(
        type=ComponentType.ICON,
        category=ComponentCategory.CONTENT,
        description="Small symbolic graphic or glyph",
        aliases=("glyph", "symbol", "icon_button"),
        html_equivalents=("i", "svg.icon", "span.icon"),
        android_classes=("android.widget.ImageView",),
        constraints=_LEAF_CONSTRAINTS,
        examples=("Menu icon", "Status indicator", "Action symbol"),
    ),
    ComponentType.ADVERTISEMENT: ComponentMeta(
        type=ComponentType.ADVERTISEMENT,
        category=ComponentCategory.CONTENT,
        description="Promotional or advertisement content block",
        aliases=("ad", "banner_ad", "promo", "sponsored"),
        html_equivalents=("div.ad", "aside.advertisement"),
        android_classes=("com.google.android.gms.ads.AdView",),
        constraints=_LEAF_CONSTRAINTS,
        examples=("Banner ad", "Sponsored content", "Promotional tile"),
    ),
    # === CONTROLS ===
    ComponentType.BUTTON: ComponentMeta(
        type=ComponentType.BUTTON,
        category=ComponentCategory.CONTROL,
        description="Clickable action trigger with visual feedback",
        aliases=("btn", "action", "cta", "submit"),
        html_equivalents=("button", "input[type=submit]", "input[type=button]"),
        android_classes=("android.widget.Button",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Submit button", "Cancel action", "Add to cart"),
    ),
    ComponentType.TEXT_BUTTON: ComponentMeta(
        type=ComponentType.TEXT_BUTTON,
        category=ComponentCategory.CONTROL,
        description="Text-styled clickable action without heavy visual treatment",
        aliases=("link_button", "flat_button", "text_link"),
        html_equivalents=("a", "button.text"),
        android_classes=("android.widget.Button",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Learn more", "See all", "Forgot password?"),
    ),
    ComponentType.INPUT: ComponentMeta(
        type=ComponentType.INPUT,
        category=ComponentCategory.CONTROL,
        description="Single-line text entry field",
        aliases=("textfield", "text_input", "field", "edittext"),
        html_equivalents=(
            "input[type=text]",
            "input[type=email]",
            "input[type=password]",
        ),
        android_classes=("android.widget.EditText",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Username field", "Search box", "Email input"),
    ),
    ComponentType.CHECKBOX: ComponentMeta(
        type=ComponentType.CHECKBOX,
        category=ComponentCategory.CONTROL,
        description="Binary toggle for multiple selection options",
        aliases=("check", "tick", "checkmark"),
        html_equivalents=("input[type=checkbox]",),
        android_classes=("android.widget.CheckBox",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Terms acceptance", "Feature toggles", "Multi-select options"),
    ),
    ComponentType.RADIO_BUTTON: ComponentMeta(
        type=ComponentType.RADIO_BUTTON,
        category=ComponentCategory.CONTROL,
        description="Single selection from mutually exclusive options",
        aliases=("radio", "option"),
        html_equivalents=("input[type=radio]",),
        android_classes=("android.widget.RadioButton",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Gender selection", "Payment method", "Shipping option"),
    ),
    ComponentType.SWITCH: ComponentMeta(
        type=ComponentType.SWITCH,
        category=ComponentCategory.CONTROL,
        description="Binary on/off toggle control",
        aliases=("toggle", "on_off", "flip"),
        html_equivalents=("input[type=checkbox].switch",),
        android_classes=(
            "android.widget.Switch",
            "androidx.appcompat.widget.SwitchCompat",
        ),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Dark mode toggle", "Notification setting", "Feature flag"),
    ),
    ComponentType.SLIDER: ComponentMeta(
        type=ComponentType.SLIDER,
        category=ComponentCategory.CONTROL,
        description="Continuous value selection along a track",
        aliases=("range", "seekbar", "track"),
        html_equivalents=("input[type=range]",),
        android_classes=("android.widget.SeekBar",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Volume control", "Brightness setting", "Price range"),
    ),
    ComponentType.SPINNER: ComponentMeta(
        type=ComponentType.SPINNER,
        category=ComponentCategory.CONTROL,
        description="Dropdown selector for single value from list",
        aliases=("dropdown", "select", "picker", "combobox"),
        html_equivalents=("select",),
        android_classes=("android.widget.Spinner",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Country selector", "Category filter", "Sort order"),
    ),
    ComponentType.DATE_PICKER: ComponentMeta(
        type=ComponentType.DATE_PICKER,
        category=ComponentCategory.CONTROL,
        description="Calendar-based date selection control",
        aliases=("calendar", "date_selector", "datepicker"),
        html_equivalents=("input[type=date]",),
        android_classes=("android.widget.DatePicker",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Birth date", "Appointment date", "Travel dates"),
    ),
    ComponentType.NUMBER_STEPPER: ComponentMeta(
        type=ComponentType.NUMBER_STEPPER,
        category=ComponentCategory.CONTROL,
        description="Increment/decrement control for numeric values",
        aliases=("stepper", "counter", "quantity", "incrementer"),
        html_equivalents=("input[type=number]",),
        android_classes=("android.widget.NumberPicker",),
        constraints=_CONTROL_CONSTRAINTS,
        examples=("Quantity selector", "Guest count", "Item amount"),
    ),
    ComponentType.DIVIDER: ComponentMeta(
        type=ComponentType.DIVIDER,
        category=ComponentCategory.CONTENT,
        description="Visual separator between content",
        aliases=("separator", "hr", "line"),
        html_equivalents=("hr",),
        constraints=_CONTROL_CONSTRAINTS,  # Leaf node
    ),
    ComponentType.DATA_GRID: ComponentMeta(
        type=ComponentType.DATA_GRID,
        category=ComponentCategory.CONTAINER,
        description="Tabular data display",
        aliases=("table", "grid", "datatable"),
        html_equivalents=("table",),
        constraints=_VERTICAL_CONTAINER,  # Allows children
    ),
    ComponentType.TREE: ComponentMeta(
        type=ComponentType.TREE,
        category=ComponentCategory.NAVIGATION,
        description="Hierarchical tree view",
        aliases=("file_tree", "nested_list"),
        html_equivalents=("ul", "li"),
        constraints=_VERTICAL_CONTAINER,
    ),
    ComponentType.MENU_BAR: ComponentMeta(
        type=ComponentType.MENU_BAR,
        category=ComponentCategory.NAVIGATION,
        description="Top-level application menu",
        aliases=("menubar", "app_menu"),
        constraints=_HORIZONTAL_CONTAINER,
    ),
}


def get_component_meta(component_type: ComponentType) -> ComponentMeta:
    """Get rich metadata for a component type.

    Args:
        component_type: The component type to look up.

    Returns:
        ComponentMeta with full metadata for the type.

    Raises:
        KeyError: If component type not found in registry.
    """
    return COMPONENT_REGISTRY[component_type]


def get_component_category(component_type: ComponentType) -> ComponentCategory:
    """Get the category for a component type.

    Args:
        component_type: The component type to categorize.

    Returns:
        The ComponentCategory for this type.
    """
    return COMPONENT_REGISTRY[component_type].category


def get_components_by_category(category: ComponentCategory) -> list[ComponentType]:
    """Get all component types in a category.

    Args:
        category: The category to filter by.

    Returns:
        List of ComponentType values in the category.
    """
    return [
        meta.type for meta in COMPONENT_REGISTRY.values() if meta.category == category
    ]


def resolve_alias(alias: str) -> ComponentType | None:
    """Resolve a component alias to its canonical type.

    Args:
        alias: The alias string to resolve (case-insensitive).

    Returns:
        The canonical ComponentType, or None if not found.
    """
    alias_lower = alias.lower().strip()

    # Direct match first
    for ct in ComponentType:
        if ct.value == alias_lower:
            return ct

    # Search aliases
    for meta in COMPONENT_REGISTRY.values():
        if alias_lower in meta.aliases:
            return meta.type

    return None


def get_constraints(component_type: ComponentType) -> ComponentConstraints:
    """Get structural constraints for a component type.

    Args:
        component_type: The component type.

    Returns:
        ComponentConstraints defining structural rules.
    """
    return COMPONENT_REGISTRY[component_type].constraints


# === SCHEMA GENERATION ===


class LayoutNodeSchema(BaseModel):
    """Pydantic model for LayoutNode JSON Schema generation.

    This model mirrors the MID LayoutNode but is specifically designed
    for schema export and LLM prompt injection.
    """

    # Identity
    id: str = Field(
        ...,
        description="Unique identifier (e.g., 'header', 'main-content')",
        examples=["root", "header", "nav-bar", "content-area", "submit-btn"],
    )
    type: ComponentType = Field(
        ...,
        description="Component type from the 26-category Rico taxonomy",
    )

    # Content
    label: str | None = Field(
        None,
        description="Human-readable text content displayed by the component",
        examples=["Submit", "Welcome Back", "Search..."],
    )

    # Structure
    children: list["LayoutNodeSchema"] = Field(
        default_factory=list,
        description="Nested child nodes for hierarchical layouts",
    )

    # Layout - Flex/Grid
    flex_ratio: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Grid span ratio (1-12) for relative width in parent",
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
        description="Flow direction: horizontal, vertical, or overlay",
    )
    display: Display = Field(
        default=Display.FLEX,
        description="Layout mode: flex (default), grid, or block",
    )
    grid_columns: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Number of grid columns (only when display=grid)",
    )
    grid_rows: int | None = Field(
        default=None,
        ge=1,
        description="Number of grid rows (only when display=grid)",
    )

    # Layout - Flex Container behavior
    align: Alignment | None = Field(
        default=None,
        description="Cross-axis alignment for children (align-items)",
    )
    justify: Justify | None = Field(
        default=None,
        description="Main-axis distribution for children (justify-content)",
    )
    align_content: AlignContent | None = Field(
        default=None,
        description="Multi-line cross-axis alignment (align-content)",
    )
    gap: int | None = Field(
        default=None,
        ge=0,
        description="Spacing between children in pixels",
    )
    wrap: Wrap | None = Field(
        default=None,
        description="Overflow wrap behavior",
    )
    padding: int | None = Field(
        default=None,
        ge=0,
        description="Internal padding in pixels",
    )

    # Layout - Flex Item behavior
    align_self: AlignSelf | None = Field(
        default=None,
        description="Self alignment override (align-self)",
    )

    # Text Styling
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

    model_config = {"use_enum_values": True}


def export_json_schema() -> dict[str, Any]:
    """Export the complete LayoutNode JSON Schema.

    Returns:
        JSON Schema dict suitable for validation or LLM prompts.
    """
    return LayoutNodeSchema.model_json_schema()


def export_component_enum_schema() -> dict[str, Any]:
    """Export schema for ComponentType enum with descriptions.

    Returns:
        Dict mapping component values to their descriptions.
    """
    return {ct.value: COMPONENT_REGISTRY[ct].description for ct in ComponentType}


def export_llm_schema() -> dict[str, Any]:
    """Export an LLM-optimized schema with examples and constraints.

    This schema is designed for injection into LLM prompts to guide
    structured JSON output generation.

    Returns:
        Dict with schema, component descriptions, and usage examples.
    """
    base_schema = export_json_schema()

    return {
        "schema": base_schema,
        "component_types": export_component_enum_schema(),
        "categories": {
            cat.value: [ct.value for ct in get_components_by_category(cat)]
            for cat in ComponentCategory
        },
        "orientation_options": [o.value for o in Orientation],
        "constraints": {
            "flex_ratio": {"min": 1, "max": 12, "description": "12-column grid system"},
            "id": "Must be unique within the tree",
        },
        "examples": {
            "simple_form": {
                "id": "login-form",
                "type": "container",
                "orientation": "vertical",
                "children": [
                    {"id": "title", "type": "text", "label": "Sign In"},
                    {"id": "email", "type": "input", "label": "Email"},
                    {"id": "password", "type": "input", "label": "Password"},
                    {"id": "submit", "type": "button", "label": "Sign In"},
                ],
            },
            "nav_bar": {
                "id": "header",
                "type": "toolbar",
                "orientation": "horizontal",
                "children": [
                    {"id": "menu-icon", "type": "icon", "flex_ratio": 1},
                    {
                        "id": "title",
                        "type": "text",
                        "label": "My App",
                        "flex_ratio": 10,
                    },
                    {"id": "search-icon", "type": "icon", "flex_ratio": 1},
                ],
            },
        },
    }


# === SCHEMA VALIDATION ===


@dataclass
class SchemaValidationError:
    """Represents a schema validation error."""

    path: str
    message: str
    error_type: str


def validate_component_type(value: str) -> ComponentType | None:
    """Validate and convert a string to ComponentType.

    Args:
        value: String value to validate.

    Returns:
        ComponentType if valid, None otherwise.
    """
    try:
        return ComponentType(value)
    except ValueError:
        return resolve_alias(value)


def validate_orientation(value: str) -> Orientation | None:
    """Validate and convert a string to Orientation.

    Args:
        value: String value to validate.

    Returns:
        Orientation if valid, None otherwise.
    """
    try:
        return Orientation(value)
    except ValueError:
        return None


def validate_layout_dict(
    data: dict[str, Any], path: str = "root"
) -> list[SchemaValidationError]:
    """Validate a layout dictionary against the schema.

    Args:
        data: Dictionary representing a layout node.
        path: Current path in the tree for error reporting.

    Returns:
        List of validation errors found.
    """
    errors: list[SchemaValidationError] = []

    # Required fields
    if "id" not in data:
        errors.append(
            SchemaValidationError(path, "Missing required field 'id'", "missing_field")
        )
    if "type" not in data:
        errors.append(
            SchemaValidationError(
                path, "Missing required field 'type'", "missing_field"
            )
        )
    else:
        if validate_component_type(data["type"]) is None:
            errors.append(
                SchemaValidationError(
                    path,
                    f"Invalid component type: {data['type']}",
                    "invalid_enum",
                )
            )

    # Optional fields with constraints
    if "flex_ratio" in data:
        fr = data["flex_ratio"]
        if not isinstance(fr, int) or fr < 1 or fr > 12:
            errors.append(
                SchemaValidationError(
                    path,
                    f"flex_ratio must be integer 1-12, got: {fr}",
                    "invalid_range",
                )
            )

    if "orientation" in data:
        if validate_orientation(data["orientation"]) is None:
            errors.append(
                SchemaValidationError(
                    path,
                    f"Invalid orientation: {data['orientation']}",
                    "invalid_enum",
                )
            )

    # Validate children recursively
    if "children" in data:
        if not isinstance(data["children"], list):
            errors.append(
                SchemaValidationError(path, "children must be a list", "invalid_type")
            )
        else:
            seen_ids: set[str] = set()
            for i, child in enumerate(data["children"]):
                if not isinstance(child, dict):
                    errors.append(
                        SchemaValidationError(
                            f"{path}.children[{i}]",
                            "Child must be an object",
                            "invalid_type",
                        )
                    )
                else:
                    # Check for duplicate IDs
                    child_id = child.get("id")
                    if child_id and child_id in seen_ids:
                        errors.append(
                            SchemaValidationError(
                                f"{path}.children[{i}]",
                                f"Duplicate id: {child_id}",
                                "duplicate_id",
                            )
                        )
                    if child_id:
                        seen_ids.add(child_id)

                    # Recurse
                    errors.extend(validate_layout_dict(child, f"{path}.children[{i}]"))

    return errors


def is_valid_layout_dict(data: dict[str, Any]) -> bool:
    """Check if a layout dictionary is valid.

    Args:
        data: Dictionary to validate.

    Returns:
        True if valid, False otherwise.
    """
    return len(validate_layout_dict(data)) == 0


__all__ = [
    # Enums
    "ComponentCategory",
    "ComponentType",
    "ComponentType",
    "Orientation",
    "Alignment",
    "Justify",
    "Wrap",
    "Display",
    "AlignContent",
    "AlignSelf",
    "TextSize",
    "TextWeight",
    "TextTransform",
    "TextAlign",
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
