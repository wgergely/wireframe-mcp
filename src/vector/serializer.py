"""Schema-aware LayoutNode serializer for embedding generation.

Converts LayoutNode trees into embedding-optimized text representations
that capture structural hierarchy, component semantics, and text content.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator

from src.mid import LayoutNode, Orientation
from src.schema import ComponentType, get_component_category


class SerializationFormat(str, Enum):
    """Available serialization output formats."""

    INDENTED = "indented"  # Human-readable tree with indentation
    LINEARIZED = "linearized"  # Compact sequence with XML-like tags


@dataclass
class SerializationConfig:
    """Configuration for serialization behavior.

    Attributes:
        format: Output format style.
        include_category: Include component category prefix [CATEGORY:type].
        include_orientation: Include @orientation marker.
        include_flex: Include flex:N ratio marker.
        max_label_length: Maximum label length before truncation.
        indent_size: Spaces per indentation level (INDENTED format).
    """

    format: SerializationFormat = SerializationFormat.INDENTED
    include_category: bool = True
    include_orientation: bool = True
    include_flex: bool = True
    max_label_length: int = 50
    indent_size: int = 2


@dataclass
class SerializedLayout:
    """Result of serialization with metadata.

    Attributes:
        id: Original item identifier.
        source: Provider name (rico, egfe, etc.).
        dataset: Dataset variant.
        text: Serialized text output.
        node_count: Total nodes in tree.
        max_depth: Maximum nesting depth.
        component_summary: Component type counts.
    """

    id: str
    source: str
    dataset: str
    text: str
    node_count: int
    max_depth: int
    component_summary: dict[str, int] = field(default_factory=dict)


class LayoutSerializer:
    """Schema-aware serializer for LayoutNode trees.

    Converts LayoutNode trees into text formats optimized for embedding
    models. Leverages the schema module's COMPONENT_REGISTRY for semantic
    enrichment of component types.

    Example:
        >>> serializer = LayoutSerializer()
        >>> text = serializer.serialize(layout_node)
        >>> print(text)
        [CONTAINER:container] root @vertical
          [CONTROL:button] submit "Click Me"
    """

    def __init__(self, config: SerializationConfig | None = None):
        """Initialize serializer with configuration.

        Args:
            config: Serialization options. Uses defaults if None.
        """
        self.config = config or SerializationConfig()

    def serialize(self, node: LayoutNode) -> str:
        """Serialize a LayoutNode tree to text.

        Args:
            node: Root LayoutNode to serialize.

        Returns:
            Text representation of the layout tree.
        """
        if self.config.format == SerializationFormat.INDENTED:
            lines = self._serialize_indented(node, depth=0)
            return "\n".join(lines)
        else:
            return self._serialize_linearized(node)

    def serialize_with_metadata(
        self,
        node: LayoutNode,
        item_id: str,
        source: str,
        dataset: str,
    ) -> SerializedLayout:
        """Serialize with full metadata for indexing.

        Args:
            node: Root LayoutNode to serialize.
            item_id: Unique identifier for this item.
            source: Provider name.
            dataset: Dataset variant.

        Returns:
            SerializedLayout with text and statistics.
        """
        text = self.serialize(node)
        node_count = self._count_nodes(node)
        max_depth = self._max_depth(node)
        component_summary = self._count_components(node)

        return SerializedLayout(
            id=item_id,
            source=source,
            dataset=dataset,
            text=text,
            node_count=node_count,
            max_depth=max_depth,
            component_summary=component_summary,
        )

    def serialize_batch(
        self,
        items: Iterator[tuple[LayoutNode, str, str, str]],
        max_workers: int = 4,
        on_complete: Callable[[SerializedLayout], None] | None = None,
    ) -> Iterator[SerializedLayout]:
        """Batch serialize with multi-threaded processing.

        Args:
            items: Iterator of (node, id, source, dataset) tuples.
            max_workers: Number of concurrent threads.
            on_complete: Optional callback for each completed item.

        Yields:
            SerializedLayout results as they complete.
        """
        # Collect items for submission (can't iterate twice)
        item_list = list(items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(
                    self.serialize_with_metadata, node, item_id, source, dataset
                ): (item_id,)
                for node, item_id, source, dataset in item_list
            }

            for future in as_completed(future_to_item):
                result = future.result()
                if on_complete:
                    on_complete(result)
                yield result

    def _serialize_indented(self, node: LayoutNode, depth: int) -> list[str]:
        """Serialize node to indented format lines.

        Args:
            node: Current node to serialize.
            depth: Current indentation depth.

        Returns:
            List of text lines for this node and children.
        """
        lines = []
        indent = " " * (depth * self.config.indent_size)

        # Build node representation
        parts = []

        # Get type value (handle both enum and string)
        node_type = node.type
        if isinstance(node_type, ComponentType):
            type_value = node_type.value
        else:
            type_value = str(node_type)

        # Category and type
        if self.config.include_category:
            try:
                category = get_component_category(ComponentType(type_value))
                parts.append(f"[{category.value.upper()}:{type_value}]")
            except (ValueError, KeyError):
                parts.append(f"[{type_value}]")
        else:
            parts.append(f"[{type_value}]")

        # ID
        parts.append(node.id)

        # Orientation (only if non-default vertical)
        if self.config.include_orientation:
            orientation = node.orientation
            if isinstance(orientation, Orientation):
                orient_value = orientation.value
            else:
                orient_value = str(orientation)

            if orient_value != "vertical":
                parts.append(f"@{orient_value}")

        # Flex ratio (only if non-default 1)
        if self.config.include_flex and node.flex_ratio != 1:
            parts.append(f"flex:{node.flex_ratio}")

        # Label (truncated if needed)
        if node.label:
            label = node.label[: self.config.max_label_length]
            if len(node.label) > self.config.max_label_length:
                label += "..."
            # Escape quotes in label
            label = label.replace('"', '\\"')
            parts.append(f'"{label}"')

        lines.append(indent + " ".join(parts))

        # Recursively process children
        for child in node.children:
            lines.extend(self._serialize_indented(child, depth + 1))

        return lines

    def _serialize_linearized(self, node: LayoutNode) -> str:
        """Serialize node to linearized sequence format.

        Args:
            node: Root node to serialize.

        Returns:
            Compact string with XML-like tags.
        """
        parts = []
        self._linearize_node(node, parts)
        return " ".join(parts)

    def _linearize_node(self, node: LayoutNode, parts: list[str]) -> None:
        """Recursively linearize a node into parts list.

        Args:
            node: Current node.
            parts: Accumulator list for string parts.
        """
        # Get type value
        node_type = node.type
        if isinstance(node_type, ComponentType):
            type_value = node_type.value
        else:
            type_value = str(node_type)

        # Orientation short code
        orientation = node.orientation
        if isinstance(orientation, Orientation):
            orient_value = orientation.value
        else:
            orient_value = str(orientation)

        orient_code = {"horizontal": "H", "vertical": "V", "overlay": "O"}.get(
            orient_value, "V"
        )

        is_leaf = len(node.children) == 0

        if is_leaf:
            # Self-closing tag
            if node.label:
                label = self._escape_label(node.label)
                parts.append(f'<{node.id}:{type_value}:"{label}"/>')
            else:
                parts.append(f"<{node.id}:{type_value}/>")
        else:
            # Opening tag
            parts.append(f"<{node.id}:{type_value}:{orient_code}>")

            # Children
            for child in node.children:
                self._linearize_node(child, parts)

            # Closing tag
            parts.append(f"</{node.id}>")

    def _escape_label(self, label: str) -> str:
        """Escape special characters in label.

        Args:
            label: Raw label text.

        Returns:
            Escaped label safe for embedding in tags.
        """
        # Truncate if needed
        if len(label) > self.config.max_label_length:
            label = label[: self.config.max_label_length] + "..."

        # Escape quotes and angle brackets
        return label.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    def _count_nodes(self, node: LayoutNode) -> int:
        """Count total nodes in tree.

        Args:
            node: Root node.

        Returns:
            Total node count including root.
        """
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _max_depth(self, node: LayoutNode, current: int = 0) -> int:
        """Calculate maximum tree depth.

        Args:
            node: Current node.
            current: Current depth level.

        Returns:
            Maximum depth from this node down.
        """
        if not node.children:
            return current

        return max(self._max_depth(child, current + 1) for child in node.children)

    def _count_components(self, node: LayoutNode) -> dict[str, int]:
        """Count component types in tree.

        Args:
            node: Root node.

        Returns:
            Dict mapping component type to count.
        """
        counts: dict[str, int] = {}
        self._accumulate_components(node, counts)
        return counts

    def _accumulate_components(
        self, node: LayoutNode, counts: dict[str, int]
    ) -> None:
        """Recursively accumulate component counts.

        Args:
            node: Current node.
            counts: Accumulator dict.
        """
        node_type = node.type
        if isinstance(node_type, ComponentType):
            type_value = node_type.value
        else:
            type_value = str(node_type)

        counts[type_value] = counts.get(type_value, 0) + 1

        for child in node.children:
            self._accumulate_components(child, counts)


def serialize_layout(
    node: LayoutNode,
    format: SerializationFormat = SerializationFormat.INDENTED,
) -> str:
    """Quick serialize a layout node with defaults.

    Args:
        node: LayoutNode to serialize.
        format: Output format style.

    Returns:
        Serialized text representation.
    """
    config = SerializationConfig(format=format)
    serializer = LayoutSerializer(config)
    return serializer.serialize(node)


__all__ = [
    "SerializationFormat",
    "SerializationConfig",
    "SerializedLayout",
    "LayoutSerializer",
    "serialize_layout",
]
