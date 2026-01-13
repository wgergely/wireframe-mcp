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
        - Width hints via `width: N%`

    Example output:
        ```d2
        root: Dashboard {
          direction: right

          sidebar: Navigation {
            width: 25%
          }

          main: Content {
            width: 75%
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

        # Width hint based on flex ratio (if parent context exists)
        if node.flex_ratio != 1:
            width_pct = self._flex_to_percentage(node.flex_ratio)
            lines.append(f"{prefix}  width: {width_pct}%")

        # Recurse into children
        for child in node.children:
            lines.append("")  # Blank line between children
            lines.append(self._transpile_node(child, indent + 1))

        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    def _escape_label(self, label: str) -> str:
        """Escape special characters in D2 labels.

        Args:
            label: Raw label text.

        Returns:
            str: Escaped label safe for D2.
        """
        # D2 uses double quotes for labels with special chars
        if any(c in label for c in "{}:;|"):
            return f'"{label}"'
        return label

    def _flex_to_percentage(self, flex_ratio: int) -> int:
        """Convert flex ratio to approximate percentage.

        Uses a 12-column grid as the base.

        Args:
            flex_ratio: Flex ratio value (1-12).

        Returns:
            int: Percentage width (rounded).
        """
        return round((flex_ratio / 12) * 100)
