"""PlantUML Salt DSL provider for layout transpilation.

PlantUML Salt is a GUI mockup language with a sketchy, wireframe aesthetic.
It's ideal for low-fidelity prototypes and has high tolerance for LLM output.

See: https://plantuml.com/salt
"""

from src.mid import ComponentType, LayoutNode, Orientation
from src.providers.lib import LayoutProvider, register_provider


@register_provider
class PlantUMLProvider(LayoutProvider):
    """Transpiles LayoutNode IR to PlantUML Salt syntax.

    Salt features used:
        - `{+` for tabbed panels
        - `{#` for grid/table layouts
        - `{` for simple grouping
        - `[button]` for buttons
        - `"text"` for labels
        - `^dropdown^` for dropdowns

    Example output:
        ```plantuml
        @startsalt
        {
          {# Navigation | Content }
          { [Button 1] | [Button 2] }
        }
        @endsalt
        ```
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "plantuml"

    @property
    def file_extension(self) -> str:
        """PlantUML file extension."""
        return ".puml"

    def transpile(self, node: LayoutNode) -> str:
        """Transpile a LayoutNode tree to PlantUML Salt."""
        body = self._transpile_node(node, indent=1)
        return f"@startsalt\n{body}\n@endsalt"

    def _transpile_node(self, node: LayoutNode, indent: int) -> str:
        """Recursively transpile a node and its children.

        Args:
            node: The node to transpile.
            indent: Current indentation level.

        Returns:
            str: Salt syntax for this node.
        """
        prefix = "  " * indent

        # Determine the opening brace style based on orientation
        brace = self._get_brace_style(node)

        # Generate content for this node
        content = self._render_component(node)

        if not node.children:
            # Leaf node - just render the component
            return f"{prefix}{content}"

        # Container with children
        lines: list[str] = []
        lines.append(f"{prefix}{brace}")

        # Add node label/title if present
        if node.label:
            lines.append(f'{prefix}  "{node.label}"')

        # Render children based on orientation
        if node.orientation == Orientation.HORIZONTAL:
            # Horizontal: join children with pipe separator
            child_contents = [
                self._transpile_node(child, 0).strip() for child in node.children
            ]
            lines.append(f"{prefix}  {' | '.join(child_contents)}")
        else:
            # Vertical: stack children
            for child in node.children:
                lines.append(self._transpile_node(child, indent + 1))

        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    def _get_brace_style(self, node: LayoutNode) -> str:
        """Determine the Salt brace style for a node."""
        if node.orientation == Orientation.HORIZONTAL:
            return "{#"  # Grid layout
        if node.type in (ComponentType.NAVBAR, ComponentType.DRAWER, ComponentType.TAB_BAR):
            return "{+"  # Tabbed panel
        return "{"  # Default grouping

    def _render_component(self, node: LayoutNode) -> str:
        """Render a single component without children."""
        label = node.label or node.id

        match node.type:
            case ComponentType.BUTTON | ComponentType.TEXT_BUTTON:
                return f"[{label}]"
            case ComponentType.INPUT:
                return f'"{label}    "'  # Spaces suggest input field
            case ComponentType.CHECKBOX | ComponentType.SWITCH:
                return f"[X] {label}"
            case ComponentType.RADIO_BUTTON:
                return f"() {label}"
            case ComponentType.TEXT | ComponentType.LIST_ITEM:
                return f'"{label}"'
            case ComponentType.IMAGE:
                return f"<&image> {label}"
            case ComponentType.ICON:
                return f"<&star> {label}"
            case _:
                return f'"{label}"'
