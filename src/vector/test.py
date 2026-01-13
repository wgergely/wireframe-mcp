"""Unit tests for the Vector module."""

import numpy as np
import pytest

from src.mid import LayoutNode, Orientation
from src.schema import ComponentType
from src.vector import (
    LayoutSerializer,
    SerializationConfig,
    SerializationFormat,
    serialize_layout,
)

# Check if FAISS is available
try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")


class TestLayoutSerializer:
    """Tests for LayoutSerializer."""

    @pytest.fixture
    def simple_tree(self) -> LayoutNode:
        """Create a simple test layout tree."""
        return LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            orientation=Orientation.VERTICAL,
            children=[
                LayoutNode(
                    id="header",
                    type=ComponentType.TOOLBAR,
                    orientation=Orientation.HORIZONTAL,
                    children=[
                        LayoutNode(id="menu", type=ComponentType.ICON),
                        LayoutNode(
                            id="title",
                            type=ComponentType.TEXT,
                            label="My App",
                            flex_ratio=10,
                        ),
                    ],
                ),
                LayoutNode(
                    id="submit",
                    type=ComponentType.BUTTON,
                    label="Click Me",
                ),
            ],
        )

    @pytest.mark.unit
    def test_serialize_indented_format(self, simple_tree):
        """Indented format produces expected structure."""
        serializer = LayoutSerializer()
        result = serializer.serialize(simple_tree)

        # Check root node
        assert "[CONTAINER:container]" in result
        assert "root" in result

        # Check nested header
        assert "[NAVIGATION:toolbar]" in result
        assert "header" in result
        assert "@horizontal" in result

        # Check text with label
        assert '"My App"' in result
        assert "flex:10" in result

        # Check button
        assert "[CONTROL:button]" in result
        assert '"Click Me"' in result

    @pytest.mark.unit
    def test_serialize_linearized_format(self, simple_tree):
        """Linearized format produces compact output."""
        config = SerializationConfig(format=SerializationFormat.LINEARIZED)
        serializer = LayoutSerializer(config)
        result = serializer.serialize(simple_tree)

        # Check XML-like structure
        assert "<root:container:V>" in result
        assert "</root>" in result
        assert "<header:toolbar:H>" in result
        assert "</header>" in result
        assert '<submit:button:"Click Me"/>' in result

    @pytest.mark.unit
    def test_serialize_without_category(self, simple_tree):
        """Category prefix can be disabled."""
        config = SerializationConfig(include_category=False)
        serializer = LayoutSerializer(config)
        result = serializer.serialize(simple_tree)

        assert "[container]" in result
        assert "CONTAINER:" not in result

    @pytest.mark.unit
    def test_serialize_without_orientation(self, simple_tree):
        """Orientation marker can be disabled."""
        config = SerializationConfig(include_orientation=False)
        serializer = LayoutSerializer(config)
        result = serializer.serialize(simple_tree)

        assert "@horizontal" not in result
        assert "@vertical" not in result

    @pytest.mark.unit
    def test_serialize_without_flex(self, simple_tree):
        """Flex ratio marker can be disabled."""
        config = SerializationConfig(include_flex=False)
        serializer = LayoutSerializer(config)
        result = serializer.serialize(simple_tree)

        assert "flex:" not in result

    @pytest.mark.unit
    def test_label_truncation(self):
        """Long labels are truncated."""
        long_label = "A" * 100
        node = LayoutNode(
            id="test",
            type=ComponentType.TEXT,
            label=long_label,
        )

        config = SerializationConfig(max_label_length=20)
        serializer = LayoutSerializer(config)
        result = serializer.serialize(node)

        assert "..." in result
        assert "A" * 100 not in result

    @pytest.mark.unit
    def test_serialize_with_metadata(self, simple_tree):
        """Serialization with metadata returns statistics."""
        serializer = LayoutSerializer()
        result = serializer.serialize_with_metadata(
            simple_tree,
            item_id="test-123",
            source="rico",
            dataset="semantic",
        )

        assert result.id == "test-123"
        assert result.source == "rico"
        assert result.dataset == "semantic"
        assert result.node_count == 5
        assert result.max_depth == 2
        assert "container" in result.component_summary
        assert result.component_summary["container"] == 1
        assert result.component_summary["button"] == 1

    @pytest.mark.unit
    def test_convenience_function(self, simple_tree):
        """serialize_layout convenience function works."""
        result = serialize_layout(simple_tree)
        assert "[CONTAINER:container]" in result

        result_linear = serialize_layout(
            simple_tree, format=SerializationFormat.LINEARIZED
        )
        assert "<root:container:V>" in result_linear


@requires_faiss
class TestFAISSIndex:
    """Tests for FAISSIndex.

    These tests require FAISS to be installed (faiss-cpu or faiss-gpu).
    They are skipped if FAISS is not available.
    """

    @pytest.fixture
    def sample_vectors(self) -> tuple[np.ndarray, list[str]]:
        """Create sample vectors for testing."""
        np.random.seed(42)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"item_{i}" for i in range(10)]
        return vectors, ids

    @pytest.fixture(autouse=True)
    def import_faiss_types(self):
        """Import FAISS types for tests."""
        from src.vector import FAISSIndex, SearchResult

        self.FAISSIndex = FAISSIndex
        self.SearchResult = SearchResult

    @pytest.mark.unit
    def test_index_creation(self):
        """Index can be created with dimension."""
        index = self.FAISSIndex(dimension=64)
        assert index.dimension == 64
        assert index.size == 0

    @pytest.mark.unit
    def test_add_vectors(self, sample_vectors):
        """Vectors can be added to index."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids)

        assert index.size == 10

    @pytest.mark.unit
    def test_add_empty(self):
        """Adding empty arrays is handled gracefully."""
        index = self.FAISSIndex(dimension=64)
        index.add(np.array([]).reshape(0, 64), [])
        assert index.size == 0

    @pytest.mark.unit
    def test_add_mismatched_lengths(self, sample_vectors):
        """Mismatched vectors and ids raises error."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)

        with pytest.raises(ValueError, match="must match"):
            index.add(vectors, ids[:5])

    @pytest.mark.unit
    def test_add_wrong_dimension(self):
        """Wrong dimension raises error."""
        index = self.FAISSIndex(dimension=64)
        vectors = np.random.randn(5, 128).astype(np.float32)
        ids = [f"item_{i}" for i in range(5)]

        with pytest.raises(ValueError, match="dimension"):
            index.add(vectors, ids)

    @pytest.mark.unit
    def test_search_empty(self):
        """Search on empty index returns empty results."""
        index = self.FAISSIndex(dimension=64)
        query = np.random.randn(64).astype(np.float32)
        results = index.search(query, k=5)

        assert results == []

    @pytest.mark.unit
    def test_search_basic(self, sample_vectors):
        """Basic search returns ranked results."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids)

        # Search with first vector (should match itself best)
        results = index.search(vectors[0], k=5)

        assert len(results) == 5
        assert results[0].rank == 0
        assert results[0].id == "item_0"  # Should match itself
        assert results[0].score >= results[1].score  # Sorted by score

    @pytest.mark.unit
    def test_search_result_structure(self, sample_vectors):
        """SearchResult has expected attributes."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids)

        results = index.search(vectors[0], k=1)
        result = results[0]

        assert isinstance(result, self.SearchResult)
        assert isinstance(result.id, str)
        assert isinstance(result.score, float)
        assert isinstance(result.rank, int)

    @pytest.mark.unit
    def test_search_k_limit(self, sample_vectors):
        """K limits number of results."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids)

        results = index.search(vectors[0], k=3)
        assert len(results) == 3

    @pytest.mark.unit
    def test_search_k_exceeds_size(self, sample_vectors):
        """K larger than index size returns all items."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids)

        results = index.search(vectors[0], k=100)
        assert len(results) == 10

    @pytest.mark.unit
    def test_clear(self, sample_vectors):
        """Clear removes all vectors."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids)
        assert index.size == 10

        index.clear()
        assert index.size == 0

    @pytest.mark.unit
    def test_save_load(self, sample_vectors, tmp_path):
        """Index can be saved and loaded."""
        vectors, ids = sample_vectors
        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids)

        # Save
        save_path = tmp_path / "test_index"
        index.save(save_path)

        # Load
        loaded = self.FAISSIndex.load(save_path)
        assert loaded.size == 10
        assert loaded.dimension == 64

        # Search should work
        results = loaded.search(vectors[0], k=1)
        assert results[0].id == "item_0"


class TestSerializationFormats:
    """Tests for different serialization format outputs."""

    @pytest.fixture
    def nested_tree(self) -> LayoutNode:
        """Create a deeply nested tree for format testing."""
        return LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(
                    id="card",
                    type=ComponentType.CARD,
                    children=[
                        LayoutNode(
                            id="form",
                            type=ComponentType.CONTAINER,
                            children=[
                                LayoutNode(
                                    id="email",
                                    type=ComponentType.INPUT,
                                    label="Email",
                                ),
                                LayoutNode(
                                    id="password",
                                    type=ComponentType.INPUT,
                                    label="Password",
                                ),
                                LayoutNode(
                                    id="submit",
                                    type=ComponentType.BUTTON,
                                    label="Login",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

    @pytest.mark.unit
    def test_indented_preserves_depth(self, nested_tree):
        """Indented format shows hierarchy through indentation."""
        result = serialize_layout(nested_tree, SerializationFormat.INDENTED)
        lines = result.split("\n")

        # Root should have no indentation
        assert lines[0].startswith("[")

        # Card should be indented
        card_line = [line for line in lines if "card" in line][0]
        assert card_line.startswith("  ")

        # Form should be more indented
        form_line = [line for line in lines if '"Email"' in line][0]
        assert form_line.startswith("      ")

    @pytest.mark.unit
    def test_linearized_has_matching_tags(self, nested_tree):
        """Linearized format has matching open/close tags."""
        result = serialize_layout(nested_tree, SerializationFormat.LINEARIZED)

        # Count opening and closing tags
        open_count = result.count("<root:")
        close_count = result.count("</root>")
        assert open_count == close_count == 1

        open_card = result.count("<card:")
        close_card = result.count("</card>")
        assert open_card == close_card == 1


class TestComponentCategories:
    """Tests for component category serialization."""

    @pytest.mark.unit
    def test_container_category(self):
        """Container types show CONTAINER category."""
        node = LayoutNode(id="test", type=ComponentType.CARD)
        result = serialize_layout(node)
        assert "[CONTAINER:card]" in result

    @pytest.mark.unit
    def test_navigation_category(self):
        """Navigation types show NAVIGATION category."""
        node = LayoutNode(id="test", type=ComponentType.TOOLBAR)
        result = serialize_layout(node)
        assert "[NAVIGATION:toolbar]" in result

    @pytest.mark.unit
    def test_content_category(self):
        """Content types show CONTENT category."""
        node = LayoutNode(id="test", type=ComponentType.IMAGE)
        result = serialize_layout(node)
        assert "[CONTENT:image]" in result

    @pytest.mark.unit
    def test_control_category(self):
        """Control types show CONTROL category."""
        node = LayoutNode(id="test", type=ComponentType.CHECKBOX)
        result = serialize_layout(node)
        assert "[CONTROL:checkbox]" in result
