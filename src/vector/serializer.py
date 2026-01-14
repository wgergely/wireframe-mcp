"""Schema-aware LayoutNode serializer for embedding generation.

Converts LayoutNode trees into embedding-optimized text representations
that capture structural hierarchy, component semantics, and text content.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator

from src.mid import LayoutNode
from src.schema import ComponentType, get_component_category

from .types import (
    DEFAULT_ORIENTATION,
    ORIENTATION_CODES,
    OrientationCode,
    get_default_workers,
)


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
            lines, _ = self._serialize_indented_with_stats(node, depth=0)
            return "\n".join(lines)
        else:
            text, _ = self._serialize_linearized_with_stats(node)
            return text

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
        # Single-pass serialization with statistics collection
        if self.config.format == SerializationFormat.INDENTED:
            lines, stats = self._serialize_indented_with_stats(node, depth=0)
            text = "\n".join(lines)
        else:
            text, stats = self._serialize_linearized_with_stats(node)

        return SerializedLayout(
            id=item_id,
            source=source,
            dataset=dataset,
            text=text,
            node_count=stats["node_count"],
            max_depth=stats["max_depth"],
            component_summary=stats["components"],
        )

    def serialize_batch(
        self,
        items: Iterator[tuple[LayoutNode, str, str, str]],
        max_workers: int | None = None,
        on_complete: Callable[[SerializedLayout], None] | None = None,
    ) -> Iterator[SerializedLayout]:
        """Batch serialize with multi-threaded processing.

        Args:
            items: Iterator of (node, id, source, dataset) tuples.
            max_workers: Number of concurrent threads. Defaults to CPU count.
            on_complete: Optional callback for each completed item.

        Yields:
            SerializedLayout results as they complete.
        """
        if max_workers is None:
            max_workers = get_default_workers()
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

    def _serialize_indented_with_stats(
        self, node: LayoutNode, depth: int
    ) -> tuple[list[str], dict]:
        """Serialize node to indented format with single-pass statistics.

        Args:
            node: Current node to serialize.
            depth: Current indentation depth.

        Returns:
            Tuple of (lines, stats_dict) where stats contains node_count,
            max_depth, and components dict.
        """
        lines = []
        indent = " " * (depth * self.config.indent_size)
        parts = []

        # LayoutNode uses use_enum_values=True, so type/orientation are strings
        type_value = str(node.type)
        orientation = str(node.orientation)

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
        if self.config.include_orientation and orientation != DEFAULT_ORIENTATION:
            parts.append(f"@{orientation}")

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

        # Initialize stats for this node
        node_count = 1
        max_depth = depth
        components: dict[str, int] = {type_value: 1}

        # Recursively process children and accumulate stats
        for child in node.children:
            child_lines, child_stats = self._serialize_indented_with_stats(
                child, depth + 1
            )
            lines.extend(child_lines)

            # Merge stats
            node_count += child_stats["node_count"]
            max_depth = max(max_depth, child_stats["max_depth"])
            for comp_type, count in child_stats["components"].items():
                components[comp_type] = components.get(comp_type, 0) + count

        stats = {
            "node_count": node_count,
            "max_depth": max_depth,
            "components": components,
        }
        return lines, stats

    def _serialize_linearized_with_stats(self, node: LayoutNode) -> tuple[str, dict]:
        """Serialize node to linearized format with single-pass statistics.

        Args:
            node: Root node to serialize.

        Returns:
            Tuple of (text, stats_dict) where stats contains node_count,
            max_depth, and components dict.
        """
        parts: list[str] = []
        stats: dict = {"node_count": 0, "max_depth": 0, "components": {}}
        self._linearize_node(node, parts, stats, depth=0)
        return " ".join(parts), stats

    def _linearize_node(
        self, node: LayoutNode, parts: list[str], stats: dict, depth: int
    ) -> None:
        """Recursively linearize a node while collecting statistics.

        Args:
            node: Current node.
            parts: Accumulator list for string parts.
            stats: Mutable stats dict to update.
            depth: Current tree depth.
        """
        # LayoutNode uses use_enum_values=True, so type/orientation are strings
        type_value = str(node.type)

        # Update stats
        stats["node_count"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        stats["components"][type_value] = stats["components"].get(type_value, 0) + 1

        # Orientation short code
        orient_code = ORIENTATION_CODES.get(
            str(node.orientation), OrientationCode.VERTICAL.value
        )

        if not node.children:
            # Self-closing tag (leaf node)
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
                self._linearize_node(child, parts, stats, depth + 1)

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
