"""PlantUML Salt DSL provider for layout transpilation.

PlantUML Salt is a GUI mockup language with a sketchy, wireframe aesthetic.
It's ideal for low-fidelity prototypes and has high tolerance for LLM output.

See: https://plantuml.com/salt
"""

from src.mid import (
    ComponentType,
    LayoutNode,
    Orientation,
    SemanticColor,
    TextSize,
    TextTransform,
    TextWeight,
)
from src.providers.lib import LayoutProvider, register_provider


@register_provider
class PlantUMLProvider(LayoutProvider):
    """Transpiles LayoutNode IR to PlantUML Salt syntax."""

    @property
    def name(self) -> str:
        return "plantuml"

    @property
    def file_extension(self) -> str:
        return ".puml"

    def transpile(self, node: LayoutNode) -> str:
        body = self._transpile_node(node, indent=1)
        return f"@startsalt\n{body}\n@endsalt"

    def _transpile_tree_structure(self, root: LayoutNode, indent: int) -> str:
        """Render a recursive tree structure (Salt {T or {+)."""
        prefix = "  " * indent
        # Use {T for explicit tree, or {+
        lines = [f"{prefix}{{+"]

        def recurse(node: LayoutNode, depth: int):
            for child in node.children or []:
                label = child.label or child.id
                # Salt Tree uses + for hierarchy
                # e.g. + Root, ++ Child
                lines.append(f"{prefix}  {'+' * depth} {label}")
                recurse(child, depth + 1)

        recurse(root, 1)
        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    def _transpile_node(self, node: LayoutNode, indent: int) -> str:
        prefix = "  " * indent
        brace = self._get_brace_style(node)

        # Special handling for Tree structures (recursive text hierarchy)
        if node.type == ComponentType.TREE:
            return self._transpile_tree_structure(node, indent)

        content = self._render_component(node)

        if not node.children:
            return f"{prefix}{content}"

        lines: list[str] = []
        lines.append(f"{prefix}{brace}")

        if node.label:
            cleaned_label = self._apply_styling(node.label, node)
            lines.append(f'{prefix}  "{cleaned_label}"')

        # Transpile children
        child_nodes = [
            self._transpile_node(
                child, indent + (0 if node.orientation == Orientation.HORIZONTAL else 1)
            )
            for child in node.children
        ]

        # Handle GAP by inserting spacers
        if node.gap and node.gap > 0 and len(child_nodes) > 1:
            # For horizontal table {# a | b }, if we want gap we need
            # {# a | . | b }

            if node.orientation == Orientation.HORIZONTAL:
                # Interleave spacer
                new_children = []
                for i, c in enumerate(child_nodes):
                    new_children.append(c.strip())
                    if i < len(child_nodes) - 1:
                        new_children.append(".")

                lines.append(f"{prefix}  {' | '.join(new_children)}")
            else:
                # Vertical
                for i, c in enumerate(child_nodes):
                    lines.append(c)
                    if i < len(child_nodes) - 1:
                        lines.append(f'{prefix}  "."')  # Vertical spacer text
        else:
            # Standard rendering without explicit gap insertion
            if node.orientation == Orientation.HORIZONTAL:
                child_contents = [c.strip() for c in child_nodes]
                lines.append(f"{prefix}  {' | '.join(child_contents)}")
            else:
                lines += child_nodes

        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    def _get_brace_style(self, node: LayoutNode) -> str:
        # 1. Properties overriding structure
        if node.scrollable:
            return "{S"

        # 2. Specific Component Types
        if node.type == ComponentType.TREE:
            return "{+"
        if node.type == ComponentType.MENU_BAR:
            return "{*"
        if node.type in (ComponentType.MODAL, ComponentType.WEB_VIEW):
            return "{*"  # Window frame

        # 3. Enhanced Containers
        if node.type == ComponentType.CONTAINER and node.label:
            return "{^"  # Titled Group Box

        # 4. Data Grid
        if node.type == ComponentType.DATA_GRID:
            return "{#"

        # 5. Orientation Defaults
        if node.orientation == Orientation.HORIZONTAL:
            return "{#"

        if node.type in (
            ComponentType.NAVBAR,
            ComponentType.DRAWER,
            ComponentType.TAB_BAR,
        ):
            return "{+"

        return "{"

    def _apply_styling(self, text: str, node: LayoutNode) -> str:
        """Apply transform, weight, size, and color to text."""
        # 1. Transform
        if node.text_transform == TextTransform.UPPERCASE:
            text = text.upper()
        elif node.text_transform == TextTransform.LOWERCASE:
            text = text.lower()
        elif node.text_transform == TextTransform.CAPITALIZE:
            text = text.title()

        # 2. Weight
        if node.text_weight == TextWeight.BOLD:
            text = f"<b>{text}</b>"

        # 3. Size (Creole <size:N>)
        if node.text_size == TextSize.TITLE:
            text = f"<size:18>{text}</size>"
        elif node.text_size == TextSize.HEADING:
            text = f"<size:14>{text}</size>"
        elif node.text_size == TextSize.CAPTION:
            text = f"<size:10>{text}</size>"

        # 4. Color (Creole <color:Name>)
        if node.semantic_color:
            color_map = {
                SemanticColor.PRIMARY: "Blue",
                SemanticColor.SECONDARY: "Gray",
                SemanticColor.SUCCESS: "Green",
                SemanticColor.WARNING: "Orange",
                SemanticColor.DANGER: "Red",
                SemanticColor.INFO: "CornflowerBlue",
                SemanticColor.DEFAULT: "Black",
            }
            if c := color_map.get(node.semantic_color):
                text = f"<color:{c}>{text}</color>"

        return text

    def _render_component(self, node: LayoutNode) -> str:
        """Render a single component without children."""
        label = node.label or node.id

        # Apply styling to label BEFORE wrapping in component syntax
        styled = self._apply_styling(label, node)

        match node.type:
            case ComponentType.BUTTON | ComponentType.TEXT_BUTTON:
                return f"[{styled}]"
            case ComponentType.INPUT:
                return f'"{styled}    "'
            case ComponentType.CHECKBOX | ComponentType.SWITCH:
                return f"[X] {styled}"
            case ComponentType.RADIO_BUTTON:
                return f"() {styled}"
            case ComponentType.SPINNER:
                return f"^{styled}^"
            case ComponentType.SLIDER:
                return f"{{- {styled} }}"
            case ComponentType.NUMBER_STEPPER:
                return f"{{ [ - ] | {styled} | [ + ] }}"
            case ComponentType.DATE_PICKER:
                return f'"{styled}  <&calendar>"'
            case ComponentType.DIVIDER:
                return ".." if node.orientation == Orientation.HORIZONTAL else "=="
            case (
                ComponentType.TEXT
                | ComponentType.LIST_ITEM
                | ComponentType.ADVERTISEMENT
            ):
                return f'"{styled}"'
            case ComponentType.IMAGE:
                return f"<&image> {styled}"
            case ComponentType.ICON:
                return f"<&star> {styled}"
            case _:
                return f'"{styled}"'
