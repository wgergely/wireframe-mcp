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
from src.providers.lib import LayoutFeature, LayoutProvider, register_provider


@register_provider
class PlantUMLProvider(LayoutProvider):
    """Transpiles LayoutNode IR to PlantUML Salt syntax."""

    @property
    def name(self) -> str:
        return "plantuml"

    @property
    def file_extension(self) -> str:
        return ".puml"

    @property
    def supported_formats(self) -> frozenset[str]:
        """PlantUML via Kroki supports PNG, SVG, PDF, and JPEG."""
        return frozenset({"png", "svg", "pdf", "jpeg"})

    @property
    def supported_features(self) -> frozenset[LayoutFeature]:
        """PlantUML Salt supported layout features.

        PlantUML Salt supports:
        - Flex display mode (via brace syntax)
        - Horizontal/Vertical orientation (via {# and {)
        - Gap (via spacer cells)
        - Text styling (Creole markup for size, weight, color, transform)
        - Scrollable containers ({S, {SI, {S-})

        PlantUML Salt does NOT support:
        - Grid display mode (no CSS Grid equivalent)
        - Overlay orientation
        - flex_ratio (no proportional column widths)
        - Fixed width/height
        - Alignment/justify properties
        - Padding (only global skinparam)
        - Wrap
        """
        return frozenset(
            {
                LayoutFeature.DISPLAY_FLEX,
                LayoutFeature.DISPLAY_BLOCK,
                LayoutFeature.ORIENTATION_HORIZONTAL,
                LayoutFeature.ORIENTATION_VERTICAL,
                LayoutFeature.GAP,
                LayoutFeature.TEXT_SIZE,
                LayoutFeature.TEXT_WEIGHT,
                LayoutFeature.TEXT_TRANSFORM,
                LayoutFeature.SEMANTIC_COLOR,
                LayoutFeature.SCROLLABLE,
            }
        )

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
        """Transpile a node and its children to Salt syntax.

        Salt layout rules:
        - Horizontal: children on same line with | separators: {# a | b | c }
        - Vertical: children on separate lines, indented: { a \\n b \\n c }
        """
        prefix = "  " * indent
        brace = self._get_brace_style(node)

        # Special handling for Tree structures (recursive text hierarchy)
        if node.type == ComponentType.TREE:
            return self._transpile_tree_structure(node, indent)

        content = self._render_component(node)

        # Leaf node - no children, just render the component
        if not node.children:
            return f"{prefix}{content}"

        lines: list[str] = [f"{prefix}{brace}"]

        # Add label if present
        if node.label:
            styled_label = self._apply_styling(node.label, node)
            lines.append(f'{prefix}  "{styled_label}"')

        # Render children based on orientation
        if node.orientation == Orientation.HORIZONTAL:
            self._render_horizontal_children(node, lines, prefix)
        else:
            self._render_vertical_children(node, lines, indent + 1)

        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    def _render_horizontal_children(
        self, node: LayoutNode, lines: list[str], prefix: str
    ) -> None:
        """Render children horizontally with | separators.

        Salt horizontal syntax: {# item1 | item2 | item3 }
        For gap, insert spacer dots: {# item1 | . | item2 }
        """
        # Horizontal children are rendered inline, indent doesn't matter
        child_contents = [
            self._transpile_node(child, indent=0).strip() for child in node.children
        ]

        # Insert spacers for gap
        if node.gap and node.gap > 0 and len(child_contents) > 1:
            spaced = []
            for i, content in enumerate(child_contents):
                spaced.append(content)
                if i < len(child_contents) - 1:
                    spaced.append(".")
            child_contents = spaced

        lines.append(f"{prefix}  {' | '.join(child_contents)}")

    def _render_vertical_children(
        self, node: LayoutNode, lines: list[str], child_indent: int
    ) -> None:
        """Render children vertically, each on its own line.

        Salt vertical syntax: { item1 \\n item2 \\n item3 }
        For gap, insert spacer rows: { item1 \\n "." \\n item2 }
        """
        child_prefix = "  " * child_indent
        has_gap = node.gap and node.gap > 0 and len(node.children) > 1

        for i, child in enumerate(node.children):
            lines.append(self._transpile_node(child, child_indent))
            # Insert spacer between children (not after last)
            if has_gap and i < len(node.children) - 1:
                lines.append(f'{child_prefix}"."')

    def _get_brace_style(self, node: LayoutNode) -> str:
        """Determine Salt brace style based on node properties.

        Salt brace modifiers:
        - {   : Basic vertical layout (default)
        - {#  : Table/grid layout (horizontal children with | separators)
        - {+  : Tree/list layout (items prefixed with +)
        - {*  : Menu/window frame
        - {^  : Titled group box (label becomes header)
        - {S  : Scrollable pane
        - {SI : Scrollable with indicators
        - {S- : Scrollable horizontal only

        Priority: scrollable > type-specific > labeled container > orientation
        """
        # 1. Scrollable takes highest priority
        if node.scrollable:
            return "{S"

        # 2. Type-specific braces
        match node.type:
            case ComponentType.TREE:
                return "{+"
            case ComponentType.MENU_BAR:
                return "{*"
            case ComponentType.MODAL | ComponentType.WEB_VIEW:
                return "{*"  # Window frame
            case ComponentType.DATA_GRID:
                return "{#"
            case ComponentType.NAVBAR | ComponentType.DRAWER | ComponentType.TAB_BAR:
                return "{+"

        # 3. Labeled containers get group box style
        if node.type == ComponentType.CONTAINER and node.label:
            return "{^"

        # 4. Horizontal orientation uses table layout
        if node.orientation == Orientation.HORIZONTAL:
            return "{#"

        # 5. Default vertical layout
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
