"""Tests for IR layer TranspilationContext."""

import pytest

from src.ir import LayoutHint, LayoutNode, TranspilationContext
from src.mid import ComponentType, Orientation


@pytest.fixture
def sample_layout():
    """Create a sample LayoutNode for testing."""
    return LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        label="Dashboard",
        orientation=Orientation.HORIZONTAL,
        children=[
            LayoutNode(id="sidebar", type=ComponentType.DRAWER, label="Navigation"),
            LayoutNode(id="content", type=ComponentType.CONTAINER, label="Main"),
        ],
    )


class TestLayoutHint:
    """Tests for LayoutHint dataclass."""

    @pytest.mark.unit
    def test_hint_creation(self):
        """Test creating a layout hint."""
        hint = LayoutHint(
            hint_type="merge_containers",
            target_node_id="root",
            suggestion="Combine nested containers",
            confidence=0.8,
        )
        assert hint.hint_type == "merge_containers"
        assert hint.target_node_id == "root"
        assert hint.confidence == 0.8

    @pytest.mark.unit
    def test_hint_default_confidence(self):
        """Test default confidence value."""
        hint = LayoutHint(
            hint_type="simplify",
            target_node_id="node1",
            suggestion="Remove empty container",
        )
        assert hint.confidence == 0.5


class TestTranspilationContext:
    """Tests for TranspilationContext dataclass."""

    @pytest.mark.unit
    def test_context_creation(self, sample_layout):
        """Test creating a context with just a node."""
        ctx = TranspilationContext(node=sample_layout)
        assert ctx.node == sample_layout
        assert ctx.similar_layouts == []
        assert ctx.target_provider == "d2"
        assert ctx.preserve_intent is True

    @pytest.mark.unit
    def test_context_has_rag_context(self, sample_layout):
        """Test RAG context detection."""
        ctx = TranspilationContext(node=sample_layout)
        assert not ctx.has_rag_context

        similar = LayoutNode(id="similar", type=ComponentType.CONTAINER)
        ctx_with_rag = TranspilationContext(
            node=sample_layout,
            similar_layouts=[similar],
        )
        assert ctx_with_rag.has_rag_context

    @pytest.mark.unit
    def test_context_has_hints(self, sample_layout):
        """Test hints detection."""
        ctx = TranspilationContext(node=sample_layout)
        assert not ctx.has_hints

        hint = LayoutHint("opt", "node1", "do something")
        ctx_with_hints = TranspilationContext(
            node=sample_layout,
            layout_hints=[hint],
        )
        assert ctx_with_hints.has_hints

    @pytest.mark.unit
    def test_context_with_metadata(self, sample_layout):
        """Test context with metadata."""
        ctx = TranspilationContext(
            node=sample_layout,
            target_provider="plantuml",
            metadata={"source": "Rico", "item_id": "12345"},
        )
        assert ctx.target_provider == "plantuml"
        assert ctx.metadata["source"] == "Rico"

    @pytest.mark.unit
    def test_preserve_intent_flag(self, sample_layout):
        """Test preserve_intent flag."""
        ctx = TranspilationContext(
            node=sample_layout,
            preserve_intent=False,
        )
        assert ctx.preserve_intent is False
