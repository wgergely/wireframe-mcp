"""HTML to hierarchy normalizer for WebSight data.

Converts HTML DOM structures to the standardized hierarchy format
used by Rico/Enrico providers, enabling unified LayoutNode conversion.
"""

from html.parser import HTMLParser
from typing import Any

from src.mid import ComponentType

# HTML tag to ComponentType mapping
HTML_TAG_MAP: dict[str, ComponentType] = {
    # Controls
    "button": ComponentType.BUTTON,
    "input": ComponentType.INPUT,
    "textarea": ComponentType.INPUT,
    "select": ComponentType.SPINNER,
    "a": ComponentType.TEXT_BUTTON,
    # Content
    "img": ComponentType.IMAGE,
    "svg": ComponentType.ICON,
    "video": ComponentType.WEB_VIEW,
    "iframe": ComponentType.WEB_VIEW,
    "canvas": ComponentType.IMAGE,
    # Text
    "h1": ComponentType.TEXT,
    "h2": ComponentType.TEXT,
    "h3": ComponentType.TEXT,
    "h4": ComponentType.TEXT,
    "h5": ComponentType.TEXT,
    "h6": ComponentType.TEXT,
    "p": ComponentType.TEXT,
    "span": ComponentType.TEXT,
    "label": ComponentType.TEXT,
    "strong": ComponentType.TEXT,
    "em": ComponentType.TEXT,
    "b": ComponentType.TEXT,
    "i": ComponentType.TEXT,
    # Navigation
    "nav": ComponentType.NAVBAR,
    "header": ComponentType.TOOLBAR,
    "footer": ComponentType.BOTTOM_NAV,
    "menu": ComponentType.DRAWER,
    # Containers
    "div": ComponentType.CONTAINER,
    "section": ComponentType.CONTAINER,
    "article": ComponentType.CARD,
    "aside": ComponentType.CARD,
    "main": ComponentType.CONTAINER,
    "form": ComponentType.CONTAINER,
    "ul": ComponentType.CONTAINER,
    "ol": ComponentType.CONTAINER,
    "li": ComponentType.LIST_ITEM,
    "table": ComponentType.CONTAINER,
    "tr": ComponentType.LIST_ITEM,
    "td": ComponentType.CONTAINER,
    "th": ComponentType.TEXT,
}

# Tags to skip (non-visual or script elements)
SKIP_TAGS = {
    "script",
    "style",
    "meta",
    "link",
    "head",
    "title",
    "noscript",
    "template",
    "br",
    "hr",
    "wbr",
}

# Self-closing tags
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class HierarchyBuilder(HTMLParser):
    """HTML parser that builds a hierarchy dict."""

    def __init__(self):
        super().__init__()
        self.root: dict[str, Any] = {
            "tag": "root",
            "class": "html.root",
            "children": [],
            "text": "",
        }
        self.stack: list[dict] = [self.root]
        self.current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            return

        # Flush accumulated text to parent
        self._flush_text()

        attrs_dict = dict(attrs)
        node: dict[str, Any] = {
            "tag": tag,
            "class": f"html.{tag}",
            "children": [],
            "text": "",
        }

        # Extract useful attributes
        if "class" in attrs_dict:
            node["html_class"] = attrs_dict["class"]
        if "id" in attrs_dict:
            node["html_id"] = attrs_dict["id"]
        if "href" in attrs_dict:
            node["href"] = attrs_dict["href"]
        if "src" in attrs_dict:
            node["src"] = attrs_dict["src"]
        if "type" in attrs_dict:
            node["input_type"] = attrs_dict["type"]
        if "placeholder" in attrs_dict:
            node["placeholder"] = attrs_dict["placeholder"]
        if "value" in attrs_dict:
            node["value"] = attrs_dict["value"]
        if "alt" in attrs_dict:
            node["alt"] = attrs_dict["alt"]

        # Add to parent's children
        self.stack[-1]["children"].append(node)

        # Push to stack (unless void tag)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS or tag in VOID_TAGS:
            return

        # Flush accumulated text
        self._flush_text()

        # Pop from stack if matches
        if len(self.stack) > 1 and self.stack[-1]["tag"] == tag:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.current_text.append(text)

    def _flush_text(self) -> None:
        """Flush accumulated text to current node."""
        if self.current_text:
            text = " ".join(self.current_text)
            if len(self.stack) > 0:
                current = self.stack[-1]
                if current["text"]:
                    current["text"] += " " + text
                else:
                    current["text"] = text
            self.current_text = []

    def get_hierarchy(self) -> dict:
        """Get the built hierarchy."""
        self._flush_text()
        # Return body content or root
        for child in self.root["children"]:
            if child["tag"] == "body":
                return child
            if child["tag"] == "html":
                for grandchild in child["children"]:
                    if grandchild["tag"] == "body":
                        return grandchild
        # If no body found, return first meaningful child or root
        if self.root["children"]:
            return self.root["children"][0]
        return self.root


def parse_html_to_hierarchy(html: str) -> dict:
    """Parse HTML string to hierarchy dict.

    Args:
        html: HTML string to parse.

    Returns:
        Hierarchy dict compatible with Rico/Enrico format.
    """
    parser = HierarchyBuilder()
    try:
        parser.feed(html)
    except Exception:
        # Return minimal valid hierarchy on parse error
        return {"tag": "div", "class": "html.div", "children": [], "text": ""}
    return parser.get_hierarchy()


def _infer_component_type_from_html(node: dict) -> ComponentType:
    """Infer ComponentType from HTML node.

    Args:
        node: HTML hierarchy node.

    Returns:
        Inferred ComponentType.
    """
    tag = node.get("tag", "div")

    # Check for special input types
    if tag == "input":
        input_type = node.get("input_type", "text")
        if input_type == "checkbox":
            return ComponentType.CHECKBOX
        if input_type == "radio":
            return ComponentType.RADIO_BUTTON
        if input_type == "submit" or input_type == "button":
            return ComponentType.BUTTON
        if input_type == "range":
            return ComponentType.SLIDER
        return ComponentType.INPUT

    # Check for button-like classes
    html_class = node.get("html_class", "")
    if "btn" in html_class or "button" in html_class:
        return ComponentType.BUTTON
    if "card" in html_class:
        return ComponentType.CARD
    if "nav" in html_class:
        return ComponentType.NAVBAR
    if "modal" in html_class or "dialog" in html_class:
        return ComponentType.MODAL
    if "icon" in html_class:
        return ComponentType.ICON
    if "image" in html_class or "img" in html_class:
        return ComponentType.IMAGE

    # Fall back to tag mapping
    return HTML_TAG_MAP.get(tag, ComponentType.CONTAINER)


def html_hierarchy_to_layout_hierarchy(node: dict) -> dict:
    """Convert HTML hierarchy to Rico-compatible hierarchy format.

    This transforms the HTML parser output into a format that can be
    processed by the standard hierarchy_to_layout function.

    Args:
        node: HTML hierarchy node from parse_html_to_hierarchy.

    Returns:
        Rico-compatible hierarchy dict.
    """
    comp_type = _infer_component_type_from_html(node)

    result: dict[str, Any] = {
        "class": node.get("class", "html.div"),
        "componentLabel": _component_type_to_label(comp_type),
        "children": [],
    }

    # Add text if present
    text = node.get("text", "").strip()
    if text:
        result["text"] = text[:200]  # Truncate long text

    # Add placeholder as text fallback
    if not text and node.get("placeholder"):
        result["text"] = node["placeholder"]

    # Add alt text for images
    if not text and node.get("alt"):
        result["text"] = node["alt"]

    # Recursively process children
    for child in node.get("children", []):
        child_hierarchy = html_hierarchy_to_layout_hierarchy(child)
        result["children"].append(child_hierarchy)

    return result


def _component_type_to_label(comp_type: ComponentType) -> str:
    """Convert ComponentType to Rico-style componentLabel string."""
    label_map = {
        ComponentType.CONTAINER: "Container",
        ComponentType.CARD: "Card",
        ComponentType.MODAL: "Modal",
        ComponentType.WEB_VIEW: "Web View",
        ComponentType.TOOLBAR: "Toolbar",
        ComponentType.NAVBAR: "Toolbar",
        ComponentType.BOTTOM_NAV: "Bottom Navigation",
        ComponentType.DRAWER: "Drawer",
        ComponentType.TAB_BAR: "Multi-Tab",
        ComponentType.MULTI_TAB: "Multi-Tab",
        ComponentType.PAGER_INDICATOR: "Pager Indicator",
        ComponentType.TEXT: "Text",
        ComponentType.IMAGE: "Image",
        ComponentType.LIST_ITEM: "List Item",
        ComponentType.ICON: "Icon",
        ComponentType.ADVERTISEMENT: "Advertisement",
        ComponentType.BUTTON: "Text Button",
        ComponentType.TEXT_BUTTON: "Text Button",
        ComponentType.INPUT: "Input",
        ComponentType.CHECKBOX: "Checkbox",
        ComponentType.RADIO_BUTTON: "Radio Button",
        ComponentType.SWITCH: "On/Off Switch",
        ComponentType.SLIDER: "Slider",
        ComponentType.SPINNER: "Spinner",
        ComponentType.DATE_PICKER: "Date Picker",
        ComponentType.NUMBER_STEPPER: "Number Stepper",
    }
    return label_map.get(comp_type, "Container")


def normalize_html_to_hierarchy(html: str) -> dict:
    """Full pipeline: HTML string to Rico-compatible hierarchy.

    Args:
        html: Raw HTML string.

    Returns:
        Hierarchy dict ready for hierarchy_to_layout conversion.
    """
    html_hierarchy = parse_html_to_hierarchy(html)
    return html_hierarchy_to_layout_hierarchy(html_hierarchy)


__all__ = [
    "HTML_TAG_MAP",
    "parse_html_to_hierarchy",
    "html_hierarchy_to_layout_hierarchy",
    "normalize_html_to_hierarchy",
]
