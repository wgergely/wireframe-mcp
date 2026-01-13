"""Intermediate Representation (IR) layer.

The IR layer serves as the transpilation context holder, bridging the
semantic MID layer with target-specific transpiler providers.

Key components:
- TranspilationContext: Carries LayoutNode + RAG context through pipeline
- LayoutHint: Provider-specific optimization suggestions

Re-exports MID models for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Any

from src.mid import (
    COMPONENT_CATEGORIES,
    ComponentCategory,
    ComponentType,
    LayoutNode,
    Orientation,
    export_json_schema,
    get_component_category,
)


@dataclass
class LayoutHint:
    """Provider-specific layout optimization hint.

    Hints are generated from corpus analysis or RAG retrieval and
    suggest possible optimizations for the layout tree.

    Attributes:
        hint_type: Type of optimization (e.g., "merge_containers").
        target_node_id: ID of the node this hint applies to.
        suggestion: Human-readable description of the suggestion.
        confidence: Confidence score 0.0-1.0.
    """

    hint_type: str
    target_node_id: str
    suggestion: str
    confidence: float = 0.5


@dataclass
class TranspilationContext:
    """Context passed through the transpilation pipeline.

    Carries the primary LayoutNode along with RAG-derived context
    and provider-specific hints for optimization.

    Attributes:
        node: The validated LayoutNode to transpile.
        similar_layouts: Top-K similar layouts from VectorStore search.
        target_provider: Name of target transpiler (e.g., "d2", "plantuml").
        layout_hints: Optional optimization suggestions from analysis.
        preserve_intent: If True, skip aggressive layout optimizations.
        metadata: Additional context data for transpilers.
    """

    node: LayoutNode
    similar_layouts: list[LayoutNode] = field(default_factory=list)
    target_provider: str = "d2"
    layout_hints: list[LayoutHint] = field(default_factory=list)
    preserve_intent: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_rag_context(self) -> bool:
        """Check if RAG context is available."""
        return len(self.similar_layouts) > 0

    @property
    def has_hints(self) -> bool:
        """Check if layout hints are available."""
        return len(self.layout_hints) > 0


__all__ = [
    # MID re-exports (backward compatibility)
    "Orientation",
    "ComponentType",
    "ComponentCategory",
    "COMPONENT_CATEGORIES",
    "get_component_category",
    "LayoutNode",
    "export_json_schema",
    # IR-specific
    "LayoutHint",
    "TranspilationContext",
]

