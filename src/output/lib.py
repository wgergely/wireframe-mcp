"""Output formatting for layout visualization.

Generates human-readable text representations of LayoutNode trees
for user feedback and review.
"""

from dataclasses import dataclass

from src.ir import LayoutNode, TranspilationContext
from src.mid import Orientation
from src.providers import get_provider


@dataclass
class LayoutOutput:
    """Complete output for user feedback.

    Attributes:
        text_tree: Human-readable tree representation.
        dsl_code: Transpiled DSL code.
        node: Original LayoutNode.
        provider: DSL provider used.
        image_bytes: Rendered image data (optional).
        image_format: Image format (e.g., "png", "svg").
    """

    text_tree: str
    dsl_code: str
    node: LayoutNode
    provider: str
    image_bytes: bytes | None = None
    image_format: str | None = None


def format_layout_tree(node: LayoutNode, indent: int = 0) -> str:
    """Format a LayoutNode as a human-readable tree.

    Example output:
        Dashboard [container, horizontal]
        ├── Sidebar [drawer, 25%]
        │   ├── Logo [image]
        │   └── Menu [list]
        └── Main Content [container, 75%]
            └── Widget Grid [container]

    Args:
        node: Root LayoutNode to format.
        indent: Current indentation level.

    Returns:
        Formatted tree string.
    """
    lines: list[str] = []
    _format_node(node, lines, "", is_last=True, is_root=True)
    return "\n".join(lines)


def _format_node(
    node: LayoutNode,
    lines: list[str],
    prefix: str,
    is_last: bool,
    is_root: bool = False,
) -> None:
    """Recursively format a node and its children."""
    # Build connector
    if is_root:
        connector = ""
        child_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

    # Format node info
    label = node.label or node.id
    type_name = node.type if isinstance(node.type, str) else node.type.value

    # Build attributes
    attrs = [type_name]
    if node.orientation == Orientation.HORIZONTAL:
        attrs.append("horizontal")
    if node.flex_ratio != 1:
        pct = round((node.flex_ratio / 12) * 100)
        attrs.append(f"{pct}%")

    node_str = f"{label} [{', '.join(attrs)}]"
    lines.append(f"{prefix}{connector}{node_str}")

    # Format children
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        _format_node(child, lines, child_prefix, is_last_child)


class OutputGenerator:
    """Generates complete output for user feedback.

    Produces both human-readable text and DSL visualization
    from a layout node or transpilation context.
    """

    def __init__(self, default_provider: str = "d2"):
        """Initialize generator.

        Args:
            default_provider: Default DSL provider (d2, plantuml).
        """
        self._default_provider = default_provider

    def generate(
        self,
        node: LayoutNode,
        provider: str | None = None,
    ) -> LayoutOutput:
        """Generate output from a LayoutNode.

        Args:
            node: Layout to visualize.
            provider: DSL provider override.

        Returns:
            LayoutOutput with text tree and DSL code.
        """
        provider_name = provider or self._default_provider
        dsl_provider = get_provider(provider_name)

        return LayoutOutput(
            text_tree=format_layout_tree(node),
            dsl_code=dsl_provider.transpile(node),
            node=node,
            provider=provider_name,
        )

    def generate_from_context(
        self,
        context: TranspilationContext,
    ) -> LayoutOutput:
        """Generate output from TranspilationContext.

        Args:
            context: Full transpilation context.

        Returns:
            LayoutOutput with text tree and DSL code.
        """
        return self.generate(context.node, context.target_provider)


__all__ = [
    "format_layout_tree",
    "LayoutOutput",
    "OutputGenerator",
]
