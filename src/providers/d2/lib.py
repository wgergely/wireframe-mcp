"""D2 DSL provider for layout transpilation.

D2 is a declarative diagramming language with first-class support for
nested containers and grid-based layout. It's ideal for representing
hierarchical UI structures with proper spatial arrangement.

See: https://d2lang.com/
"""

from src.mid import LayoutNode, Orientation, SemanticColor, TextWeight
from src.providers.lib import LayoutFeature, LayoutProvider, register_provider
from src.schema import Display


@register_provider
class D2Provider(LayoutProvider):
    """Transpiles LayoutNode IR to D2 DSL syntax.

    D2 features used:
        - Nested containers with `{ }` blocks
        - Grid layout via `grid-columns` and `grid-rows`
        - Grid spacing via `grid-gap`
        - Text styling via `style` block (font-size, bold, font-color)

    Example output:
        ```d2
        root: Dashboard {
          grid-columns: 2
          grid-gap: 16

          sidebar: Navigation {
            style.font-size: 14
          }

          main: Content {
            grid-rows: 3
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

    @property
    def supported_features(self) -> frozenset[LayoutFeature]:
        """D2 supported layout features.

        D2 supports:
        - Flex and Grid display modes (via grid-rows/grid-columns)
        - Horizontal/Vertical orientation (via grid-columns/grid-rows)
        - Grid columns/rows
        - Gap (via grid-gap)
        - Text styling (font-size, bold, font-color)

        D2 does NOT support:
        - Overlay orientation (z-stacking)
        - flex_ratio (no proportional sizing)
        - Alignment properties (limited to 9 preset positions)
        - Padding, wrap
        - Scrollable containers
        - Text transform (uppercase/lowercase)
        """
        return frozenset(
            {
                LayoutFeature.DISPLAY_FLEX,
                LayoutFeature.DISPLAY_GRID,
                LayoutFeature.DISPLAY_BLOCK,
                LayoutFeature.ORIENTATION_HORIZONTAL,
                LayoutFeature.ORIENTATION_VERTICAL,
                LayoutFeature.GRID_COLUMNS,
                LayoutFeature.GRID_ROWS,
                LayoutFeature.GAP,
                LayoutFeature.TEXT_SIZE,
                LayoutFeature.TEXT_WEIGHT,
                LayoutFeature.SEMANTIC_COLOR,
            }
        )

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

        # Grid layout configuration
        grid_config = self._build_grid_config(node)
        for config_line in grid_config:
            lines.append(f"{prefix}  {config_line}")

        # Text styling via style block
        style_lines = self._build_style_block(node)
        for style_line in style_lines:
            lines.append(f"{prefix}  {style_line}")

        # Unsupported properties as comments (for LLM context preservation)
        hints = self._build_unsupported_hints(node)
        if hints:
            lines.append(f"{prefix}  # unsupported: {hints}")

        # Recurse into children
        for child in node.children:
            lines.append("")  # Blank line between children
            lines.append(self._transpile_node(child, indent + 1))

        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    def _build_grid_config(self, node: LayoutNode) -> list[str]:
        """Build D2 grid layout configuration.

        Uses grid-columns for horizontal layouts, grid-rows for explicit grids.
        """
        config: list[str] = []

        # Explicit grid mode
        if node.display == Display.GRID:
            if node.grid_columns:
                config.append(f"grid-columns: {node.grid_columns}")
            if node.grid_rows:
                config.append(f"grid-rows: {node.grid_rows}")
        # Horizontal orientation → use grid-columns
        elif node.orientation == Orientation.HORIZONTAL and node.children:
            config.append(f"grid-columns: {len(node.children)}")
        # Vertical with multiple children → use grid-rows for clarity
        elif node.orientation == Orientation.VERTICAL and len(node.children) > 1:
            config.append(f"grid-rows: {len(node.children)}")

        # Grid gap
        if node.gap is not None and node.gap > 0:
            config.append(f"grid-gap: {node.gap}")

        return config

    def _build_style_block(self, node: LayoutNode) -> list[str]:
        """Build D2 style properties for text styling."""
        styles: list[str] = []

        # Font size mapping
        if node.text_size:
            size_map = {
                "title": 24,
                "heading": 18,
                "body": 14,
                "caption": 11,
            }
            if size := size_map.get(
                node.text_size.value
                if hasattr(node.text_size, "value")
                else str(node.text_size)
            ):
                styles.append(f"style.font-size: {size}")

        # Bold text
        if node.text_weight == TextWeight.BOLD:
            styles.append("style.bold: true")

        # Semantic color mapping
        if node.semantic_color:
            color_map = {
                SemanticColor.PRIMARY: "#0066cc",
                SemanticColor.SECONDARY: "#666666",
                SemanticColor.SUCCESS: "#28a745",
                SemanticColor.WARNING: "#ffc107",
                SemanticColor.DANGER: "#dc3545",
                SemanticColor.INFO: "#17a2b8",
                SemanticColor.DEFAULT: "#333333",
            }
            if color := color_map.get(node.semantic_color):
                styles.append(f'style.font-color: "{color}"')

        return styles

    def _build_unsupported_hints(self, node: LayoutNode) -> str:
        """Build comment hints for unsupported properties (LLM context)."""
        hints = []

        # Alignment (not fully supported)
        if node.align:
            hints.append(f"align={node.align}")
        if node.justify:
            hints.append(f"justify={node.justify}")
        if node.align_content:
            hints.append(f"align-content={node.align_content}")
        if node.align_self:
            hints.append(f"align-self={node.align_self}")

        # Other unsupported
        if node.padding is not None:
            hints.append(f"padding={node.padding}")
        if node.wrap:
            hints.append(f"wrap={node.wrap}")
        if node.flex_ratio != 1:
            hints.append(f"flex={node.flex_ratio}")
        if node.text_transform:
            hints.append(f"text-transform={node.text_transform}")
        if node.scrollable:
            hints.append("scrollable")

        return ", ".join(hints)

    def _escape_label(self, label: str) -> str:
        """Escape D2 special chars by quoting."""
        if any(c in label for c in "{}:;|"):
            return f'"{label}"'
        return label
