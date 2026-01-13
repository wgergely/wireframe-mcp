"""Hierarchy to LayoutNode normalizer.

Converts raw UI hierarchy data from various providers (Rico, Enrico, etc.)
into standardized LayoutNode representations for the MID layer.
"""

from typing import Iterator
from uuid import uuid4

from src.mid import ComponentType, LayoutNode, Orientation

# Rico/Enrico componentLabel → ComponentType mapping
COMPONENT_LABEL_MAP: dict[str, ComponentType] = {
    # Content
    "Advertisement": ComponentType.ADVERTISEMENT,
    "Background Image": ComponentType.IMAGE,
    "Icon": ComponentType.ICON,
    "Image": ComponentType.IMAGE,
    "List Item": ComponentType.LIST_ITEM,
    "Text": ComponentType.TEXT,
    "Video": ComponentType.WEB_VIEW,  # No VIDEO type, use WEB_VIEW
    # Navigation
    "Bottom Navigation": ComponentType.BOTTOM_NAV,
    "Drawer": ComponentType.DRAWER,
    "Multi-Tab": ComponentType.MULTI_TAB,
    "Pager Indicator": ComponentType.PAGER_INDICATOR,
    "Toolbar": ComponentType.TOOLBAR,
    # Containers
    "Card": ComponentType.CARD,
    "Map View": ComponentType.WEB_VIEW,
    "Modal": ComponentType.MODAL,
    "Web View": ComponentType.WEB_VIEW,
    # Controls
    "Checkbox": ComponentType.CHECKBOX,
    "Input": ComponentType.INPUT,
    "On/Off Switch": ComponentType.SWITCH,
    "Radio Button": ComponentType.RADIO_BUTTON,
    "Slider": ComponentType.SLIDER,
    "Text Button": ComponentType.TEXT_BUTTON,
}

# Android class → ComponentType fallback mapping
ANDROID_CLASS_MAP: dict[str, ComponentType] = {
    "Button": ComponentType.BUTTON,
    "TextView": ComponentType.TEXT,
    "EditText": ComponentType.INPUT,
    "ImageView": ComponentType.IMAGE,
    "ImageButton": ComponentType.BUTTON,
    "CheckBox": ComponentType.CHECKBOX,
    "RadioButton": ComponentType.RADIO_BUTTON,
    "Switch": ComponentType.SWITCH,
    "ToggleButton": ComponentType.SWITCH,
    "SeekBar": ComponentType.SLIDER,
    "ProgressBar": ComponentType.SLIDER,
    "Spinner": ComponentType.SPINNER,
    "ListView": ComponentType.CONTAINER,
    "RecyclerView": ComponentType.CONTAINER,
    "ScrollView": ComponentType.CONTAINER,
    "ViewPager": ComponentType.MULTI_TAB,
    "TabLayout": ComponentType.TAB_BAR,
    "BottomNavigationView": ComponentType.BOTTOM_NAV,
    "Toolbar": ComponentType.TOOLBAR,
    "ActionBar": ComponentType.TOOLBAR,
    "NavigationView": ComponentType.DRAWER,
    "DrawerLayout": ComponentType.DRAWER,
    "CardView": ComponentType.CARD,
    "WebView": ComponentType.WEB_VIEW,
    "VideoView": ComponentType.WEB_VIEW,
    "MapView": ComponentType.WEB_VIEW,
    "AdView": ComponentType.ADVERTISEMENT,
}


def _get_short_class(class_name: str) -> str:
    """Extract short class name from fully qualified name."""
    if not class_name:
        return ""
    return class_name.split(".")[-1]


def _infer_component_type(node: dict) -> ComponentType:
    """Infer ComponentType from hierarchy node.

    Priority:
    1. componentLabel (Rico semantic annotation)
    2. Android class name mapping
    3. Default to CONTAINER
    """
    # Try componentLabel first (most accurate)
    label = node.get("componentLabel")
    if label and label in COMPONENT_LABEL_MAP:
        return COMPONENT_LABEL_MAP[label]

    # Try Android class mapping
    class_name = _get_short_class(node.get("class", ""))
    for class_key, comp_type in ANDROID_CLASS_MAP.items():
        if class_key in class_name:
            return comp_type

    # Default to container for nodes with children, text for leaf nodes
    if node.get("children"):
        return ComponentType.CONTAINER
    if node.get("text"):
        return ComponentType.TEXT

    return ComponentType.CONTAINER


def _calculate_flex_ratio(
    bounds: list[int] | None, parent_bounds: list[int] | None
) -> int:
    """Calculate flex ratio (1-12) based on relative width to parent."""
    if not bounds or not parent_bounds:
        return 1

    x1, y1, x2, y2 = bounds
    px1, py1, px2, py2 = parent_bounds

    parent_width = px2 - px1
    node_width = x2 - x1

    if parent_width <= 0:
        return 1

    # Calculate ratio on 12-column grid
    ratio = round((node_width / parent_width) * 12)
    return max(1, min(12, ratio))


def _infer_orientation(children: list[dict]) -> Orientation:
    """Infer orientation from children layout.

    If children are arranged more horizontally → HORIZONTAL
    If children are arranged more vertically → VERTICAL
    """
    if not children or len(children) < 2:
        return Orientation.VERTICAL

    # Check if first two children are arranged horizontally or vertically
    bounds_list = [c.get("bounds") for c in children[:3] if c.get("bounds")]
    if len(bounds_list) < 2:
        return Orientation.VERTICAL

    # Compare horizontal vs vertical distance between centers
    horizontal_spread = 0
    vertical_spread = 0

    for i in range(len(bounds_list) - 1):
        b1, b2 = bounds_list[i], bounds_list[i + 1]
        cx1 = (b1[0] + b1[2]) / 2
        cy1 = (b1[1] + b1[3]) / 2
        cx2 = (b2[0] + b2[2]) / 2
        cy2 = (b2[1] + b2[3]) / 2

        horizontal_spread += abs(cx2 - cx1)
        vertical_spread += abs(cy2 - cy1)

    return Orientation.HORIZONTAL if horizontal_spread > vertical_spread else Orientation.VERTICAL


def _generate_node_id(prefix: str, index: int) -> str:
    """Generate a unique node ID."""
    return f"{prefix}_{index}"


def hierarchy_to_layout(
    hierarchy: dict,
    parent_bounds: list[int] | None = None,
    id_prefix: str = "node",
    _counter: list[int] | None = None,
) -> LayoutNode:
    """Convert a Rico/Enrico hierarchy dict to a LayoutNode tree.

    Args:
        hierarchy: The raw hierarchy dict from Rico/Enrico.
        parent_bounds: Bounds of the parent node for flex calculation.
        id_prefix: Prefix for generated node IDs.
        _counter: Internal counter for unique ID generation.

    Returns:
        A LayoutNode tree representing the semantic UI structure.
    """
    if _counter is None:
        _counter = [0]

    node_id = _generate_node_id(id_prefix, _counter[0])
    _counter[0] += 1

    bounds = hierarchy.get("bounds")
    children_data = hierarchy.get("children", [])

    # Convert children recursively
    children: list[LayoutNode] = []
    for child in children_data:
        child_node = hierarchy_to_layout(
            child,
            parent_bounds=bounds,
            id_prefix=id_prefix,
            _counter=_counter,
        )
        children.append(child_node)

    return LayoutNode(
        id=node_id,
        type=_infer_component_type(hierarchy),
        label=hierarchy.get("text") or None,
        flex_ratio=_calculate_flex_ratio(bounds, parent_bounds),
        children=children,
        orientation=_infer_orientation(children_data),
    )


def normalize_rico_hierarchy(hierarchy: dict, item_id: str) -> LayoutNode:
    """Normalize a Rico hierarchy to LayoutNode.

    Args:
        hierarchy: Raw Rico hierarchy dict.
        item_id: The item ID (used as ID prefix).

    Returns:
        Normalized LayoutNode tree.
    """
    return hierarchy_to_layout(hierarchy, id_prefix=f"rico_{item_id}")


def normalize_enrico_hierarchy(hierarchy: dict, item_id: str) -> LayoutNode:
    """Normalize an Enrico hierarchy to LayoutNode.

    Args:
        hierarchy: Raw Enrico hierarchy dict.
        item_id: The item ID (used as ID prefix).

    Returns:
        Normalized LayoutNode tree.
    """
    return hierarchy_to_layout(hierarchy, id_prefix=f"enrico_{item_id}")


def count_components(node: LayoutNode) -> dict[str, int]:
    """Count component types in a LayoutNode tree.

    Args:
        node: The root LayoutNode.

    Returns:
        Dict mapping component type names to counts.
    """
    counts: dict[str, int] = {}

    def visit(n: LayoutNode) -> None:
        type_name = n.type.value if isinstance(n.type, ComponentType) else str(n.type)
        counts[type_name] = counts.get(type_name, 0) + 1
        for child in n.children:
            visit(child)

    visit(node)
    return counts


def extract_text_content(node: LayoutNode) -> list[str]:
    """Extract all text labels from a LayoutNode tree.

    Args:
        node: The root LayoutNode.

    Returns:
        List of non-empty text labels.
    """
    texts: list[str] = []

    def visit(n: LayoutNode) -> None:
        if n.label:
            texts.append(n.label)
        for child in n.children:
            visit(child)

    visit(node)
    return texts


def tree_depth(node: LayoutNode) -> int:
    """Calculate the depth of a LayoutNode tree.

    Args:
        node: The root LayoutNode.

    Returns:
        Maximum depth (root = 1).
    """
    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def node_count(node: LayoutNode) -> int:
    """Count total nodes in a LayoutNode tree.

    Args:
        node: The root LayoutNode.

    Returns:
        Total node count.
    """
    return 1 + sum(node_count(child) for child in node.children)


__all__ = [
    "COMPONENT_LABEL_MAP",
    "ANDROID_CLASS_MAP",
    "hierarchy_to_layout",
    "normalize_rico_hierarchy",
    "normalize_enrico_hierarchy",
    "count_components",
    "extract_text_content",
    "tree_depth",
    "node_count",
]
