"""D2 DSL provider for layout transpilation.

D2 is a declarative diagramming language with first-class support for
nested containers and constraint-based layout. It's ideal for representing
hierarchical UI structures.

See: https://d2lang.com/
"""

from src.mid import LayoutNode, Orientation
from src.providers.lib import LayoutProvider, register_provider


@register_provider
class D2Provider(LayoutProvider):
    """Transpiles LayoutNode IR to D2 DSL syntax.

    D2 features used:
        - Nested containers with `{ }` blocks
        - Direction control via `direction: right/down`
        - Width hints via comments (D2 doesn't support % widths natively)
        - Style hints via comments for align, justify, gap, padding
        - Text styling hints via comments

    Example output:
        ```d2
        root: Dashboard {
          direction: right
          # layout: align=center, justify=between, gap=16

          sidebar: Navigation {
            # width: 25%
          }

          main: Content {
            # width: 75%
          }
        }
        ```
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "d2"

    @property
    def file_extension(self) -> str:
        """D2 file extension."""
        return ".d2"

    @property
    def supported_formats(self) -> frozenset[str]:
        """D2 via Kroki only supports SVG output."""
        return frozenset({"svg"})

    def transpile(self, node: LayoutNode) -> str:
        """Transpile a LayoutNode tree to D2 DSL.

        Args:
            node: The root LayoutNode to transpile.

        Returns:
            str: Complete D2 DSL code.
        """
        return self._transpile_node(node, indent=0)

    def _transpile_node(self, node: LayoutNode, indent: int) -> str:
        """Recursively transpile a node and its children.

        Args:
            node: The node to transpile.
            indent: Current indentation level.

        Returns:
            str: D2 code for this node.
        """
        prefix = "  " * indent
        label = self._escape_label(node.label) if node.label else node.type
        lines: list[str] = []

        # Node declaration with label
        lines.append(f"{prefix}{node.id}: {label} {{")

        # Direction attribute for horizontal layouts
        if node.orientation == Orientation.HORIZONTAL:
            lines.append(f"{prefix}  direction: right")

        # Width hint based on flex ratio (as comment - D2 doesn't support % widths)
        if node.flex_ratio != 1:
            width_pct = self._flex_to_percentage(node.flex_ratio)
            lines.append(f"{prefix}  # width: {width_pct}%")

        # Layout properties as D2 style hints
        layout_hints = self._build_layout_hints(node)
        if layout_hints:
            lines.append(f"{prefix}  # layout: {layout_hints}")

        # Text styling as D2 style hints
        text_hints = self._build_text_hints(node)
        if text_hints:
            lines.append(f"{prefix}  # text: {text_hints}")

        # Recurse into children
        for child in node.children:
            lines.append("")  # Blank line between children
            lines.append(self._transpile_node(child, indent + 1))

        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    def _build_layout_hints(self, node: LayoutNode) -> str:
        """Build layout hints string from node properties."""
        hints = []
        if node.align:
            hints.append(f"align={node.align}")
        if node.justify:
            hints.append(f"justify={node.justify}")
        if node.gap is not None:
            hints.append(f"gap={node.gap}")
        if node.padding is not None:
            hints.append(f"padding={node.padding}")
        if node.wrap:
            hints.append(f"wrap={node.wrap}")
        return ", ".join(hints)

    def _build_text_hints(self, node: LayoutNode) -> str:
        """Build text styling hints string from node properties."""
        hints = []
        if node.text_size:
            hints.append(f"size={node.text_size}")
        if node.text_weight:
            hints.append(f"weight={node.text_weight}")
        if node.text_transform:
            hints.append(f"transform={node.text_transform}")
        if node.text_align:
            hints.append(f"align={node.text_align}")
        return ", ".join(hints)

    def _escape_label(self, label: str) -> str:
        """Escape D2 special chars by quoting."""
        if any(c in label for c in "{}:;|"):
            return f'"{label}"'
        return label

    def _flex_to_percentage(self, flex_ratio: int) -> int:
        """Convert flex ratio (1-12) to percentage width."""
        return round((flex_ratio / 12) * 100)
