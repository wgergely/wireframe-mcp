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


class TestSinglePassStatistics:
    """Tests for single-pass statistics optimization."""

    @pytest.fixture
    def tree_with_depth(self) -> LayoutNode:
        """Create a tree with known depth and counts."""
        return LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(
                    id="level1",
                    type=ComponentType.CARD,
                    children=[
                        LayoutNode(
                            id="level2a",
                            type=ComponentType.BUTTON,
                        ),
                        LayoutNode(
                            id="level2b",
                            type=ComponentType.BUTTON,
                        ),
                    ],
                ),
                LayoutNode(id="sibling", type=ComponentType.TEXT),
            ],
        )

    @pytest.mark.unit
    def test_single_pass_indented_stats(self, tree_with_depth):
        """Single-pass indented serialization collects correct stats."""
        serializer = LayoutSerializer()
        result = serializer.serialize_with_metadata(
            tree_with_depth,
            item_id="test",
            source="test",
            dataset="test",
        )

        assert result.node_count == 5
        assert result.max_depth == 2
        assert result.component_summary["container"] == 1
        assert result.component_summary["card"] == 1
        assert result.component_summary["button"] == 2
        assert result.component_summary["text"] == 1

    @pytest.mark.unit
    def test_single_pass_linearized_stats(self, tree_with_depth):
        """Single-pass linearized serialization collects correct stats."""
        config = SerializationConfig(format=SerializationFormat.LINEARIZED)
        serializer = LayoutSerializer(config)
        result = serializer.serialize_with_metadata(
            tree_with_depth,
            item_id="test",
            source="test",
            dataset="test",
        )

        assert result.node_count == 5
        assert result.max_depth == 2
        assert result.component_summary["container"] == 1
        assert result.component_summary["button"] == 2


class TestParallelSerialization:
    """Tests for parallel serialization with serialize_batch."""

    @pytest.fixture
    def sample_items(self) -> list[tuple[LayoutNode, str, str, str]]:
        """Create sample items for batch serialization."""
        items = []
        for i in range(10):
            node = LayoutNode(
                id=f"root_{i}",
                type=ComponentType.CONTAINER,
                children=[
                    LayoutNode(id=f"child_{i}", type=ComponentType.BUTTON),
                ],
            )
            items.append((node, f"item_{i}", "test", "sample"))
        return items

    @pytest.mark.unit
    def test_serialize_batch_returns_all_items(self, sample_items):
        """Batch serialization returns results for all items."""
        serializer = LayoutSerializer()
        results = list(serializer.serialize_batch(iter(sample_items), max_workers=2))

        assert len(results) == 10

    @pytest.mark.unit
    def test_serialize_batch_correct_ids(self, sample_items):
        """Batch serialization preserves item IDs."""
        serializer = LayoutSerializer()
        results = list(serializer.serialize_batch(iter(sample_items), max_workers=2))

        result_ids = {r.id for r in results}
        expected_ids = {f"item_{i}" for i in range(10)}
        assert result_ids == expected_ids

    @pytest.mark.unit
    def test_serialize_batch_on_complete_callback(self, sample_items):
        """Batch serialization calls on_complete callback."""
        serializer = LayoutSerializer()
        completed = []

        def on_complete(result):
            completed.append(result.id)

        list(
            serializer.serialize_batch(
                iter(sample_items), max_workers=2, on_complete=on_complete
            )
        )

        assert len(completed) == 10


@requires_faiss
class TestFAISSBatchOperations:
    """Tests for optimized FAISS batch operations."""

    @pytest.fixture
    def large_vectors(self) -> tuple[np.ndarray, list[str]]:
        """Create larger batch of vectors for testing."""
        np.random.seed(42)
        vectors = np.random.randn(100, 64).astype(np.float32)
        ids = [f"item_{i}" for i in range(100)]
        return vectors, ids

    @pytest.fixture(autouse=True)
    def import_faiss_types(self):
        """Import FAISS types for tests."""
        from src.vector import FAISSIndex

        self.FAISSIndex = FAISSIndex

    @pytest.mark.unit
    def test_add_with_normalize_false(self, large_vectors):
        """add() with normalize=False skips normalization."""
        vectors, ids = large_vectors
        # Pre-normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms

        index = self.FAISSIndex(dimension=64)
        index.add(normalized, ids, normalize=False)

        assert index.size == 100

    @pytest.mark.unit
    def test_large_batch_chunking(self):
        """Large batches are chunked efficiently."""
        np.random.seed(42)
        # Create vectors larger than OPTIMAL_ADD_BATCH_SIZE
        vectors = np.random.randn(15000, 64).astype(np.float32)
        ids = [f"item_{i}" for i in range(15000)]

        index = self.FAISSIndex(dimension=64)
        index.add(vectors, ids, batch_size=5000)

        assert index.size == 15000

        # Verify search still works
        results = index.search(vectors[0], k=1)
        assert results[0].id == "item_0"


# ============================================================================
# VectorStore Integration Tests
# ============================================================================


class MockEmbeddingBackend:
    """Mock embedding backend for testing VectorStore."""

    def __init__(self, dimension: int = 64):
        """Initialize with fixed dimension."""
        self._dimension = dimension
        self._embed_count = 0
        np.random.seed(42)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def name(self) -> str:
        """Get backend name."""
        return "mock:test"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic embeddings based on text hash."""
        self._embed_count += len(texts)
        embeddings = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            # Use hash for deterministic but unique embeddings
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embeddings[i] = np.random.randn(self._dimension).astype(np.float32)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        return self.embed([query])[0]


class MockProvider:
    """Mock corpus provider for testing."""

    def __init__(self, items: list):
        """Initialize with list of StandardizedData-like objects."""
        self._items = items
        self._name = "mock_provider"

    @property
    def name(self) -> str:
        """Get provider name."""
        return self._name

    def process(self):
        """Yield items."""
        yield from self._items


class MockStandardizedData:
    """Mock StandardizedData for testing."""

    def __init__(
        self, id: str, layout: LayoutNode, source: str = "mock", dataset: str = "test"
    ):
        """Initialize with required fields."""
        self.id = id
        self.layout = layout
        self.source = source
        self.dataset = dataset


@requires_faiss
class TestVectorStoreIntegration:
    """Integration tests for VectorStore."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock embedding backend."""
        return MockEmbeddingBackend(dimension=64)

    @pytest.fixture
    def sample_layouts(self) -> list[LayoutNode]:
        """Create sample layouts for testing."""
        return [
            LayoutNode(
                id="login_form",
                type=ComponentType.CONTAINER,
                children=[
                    LayoutNode(
                        id="email_input", type=ComponentType.INPUT, label="Email"
                    ),
                    LayoutNode(
                        id="password_input", type=ComponentType.INPUT, label="Password"
                    ),
                    LayoutNode(
                        id="login_btn", type=ComponentType.BUTTON, label="Login"
                    ),
                ],
            ),
            LayoutNode(
                id="dashboard",
                type=ComponentType.CONTAINER,
                children=[
                    LayoutNode(id="nav", type=ComponentType.TOOLBAR),
                    LayoutNode(
                        id="content",
                        type=ComponentType.CONTAINER,
                        children=[
                            LayoutNode(id="chart", type=ComponentType.IMAGE),
                            LayoutNode(id="stats_text", type=ComponentType.TEXT),
                        ],
                    ),
                ],
            ),
            LayoutNode(
                id="settings",
                type=ComponentType.CONTAINER,
                children=[
                    LayoutNode(
                        id="toggle", type=ComponentType.CHECKBOX, label="Dark Mode"
                    ),
                    LayoutNode(id="save_btn", type=ComponentType.BUTTON, label="Save"),
                ],
            ),
        ]

    @pytest.fixture
    def mock_provider(self, sample_layouts):
        """Create mock provider with sample data."""
        items = [
            MockStandardizedData(
                id=f"item_{i}",
                layout=layout,
                source="test_source",
                dataset="test_dataset",
            )
            for i, layout in enumerate(sample_layouts)
        ]
        return MockProvider(items)

    @pytest.fixture(autouse=True)
    def import_vector_store(self):
        """Import VectorStore for tests."""
        from src.vector.lib import VectorStore

        self.VectorStore = VectorStore

    @pytest.mark.unit
    def test_vectorstore_init_with_backend(self, mock_backend):
        """VectorStore initializes with custom backend."""
        store = self.VectorStore(backend=mock_backend)

        stats = store.stats()
        assert stats.total_items == 0
        assert stats.dimension == 64
        assert stats.embedding_backend == "mock:test"

    @pytest.mark.unit
    def test_index_corpus(self, mock_backend, mock_provider):
        """VectorStore indexes corpus from provider."""
        store = self.VectorStore(backend=mock_backend)

        stats = store.index_corpus(mock_provider, batch_size=10, workers=2)

        assert stats.total_items == 3
        assert stats.dimension == 64
        assert "test_source" in stats.sources
        assert stats.sources["test_source"] == 3

    @pytest.mark.unit
    def test_search_returns_results(self, mock_backend, mock_provider):
        """Search returns ranked results."""
        store = self.VectorStore(backend=mock_backend)
        store.index_corpus(mock_provider, batch_size=10, workers=2)

        results = store.search("login form with email", k=3)

        assert len(results) == 3
        assert results[0].rank == 0
        assert results[0].source == "test_source"
        assert results[0].serialized_text is not None

    @pytest.mark.unit
    def test_search_with_source_filter(self, mock_backend):
        """Search respects source filter."""
        store = self.VectorStore(backend=mock_backend)

        # Create items from two different sources
        items = [
            MockStandardizedData(
                id="item_a",
                layout=LayoutNode(id="a", type=ComponentType.BUTTON),
                source="source_a",
            ),
            MockStandardizedData(
                id="item_b",
                layout=LayoutNode(id="b", type=ComponentType.BUTTON),
                source="source_b",
            ),
        ]
        provider = MockProvider(items)
        store.index_corpus(provider, batch_size=10, workers=2)

        # Filter to source_a only
        results = store.search("button", k=5, source_filter="source_a")

        assert len(results) == 1
        assert results[0].source == "source_a"

    @pytest.mark.unit
    def test_get_metadata(self, mock_backend, mock_provider):
        """Metadata is stored and retrievable."""
        store = self.VectorStore(backend=mock_backend)
        store.index_corpus(mock_provider, batch_size=10, workers=2)

        meta = store.get_metadata("item_0")

        assert meta is not None
        assert meta["source"] == "test_source"
        assert meta["dataset"] == "test_dataset"
        assert "node_count" in meta
        assert "component_summary" in meta

    @pytest.mark.unit
    def test_save_load(self, mock_backend, mock_provider, tmp_path):
        """VectorStore saves and loads correctly."""
        store = self.VectorStore(backend=mock_backend)
        store.index_corpus(mock_provider, batch_size=10, workers=2)

        # Save (path is treated as a directory)
        save_dir = tmp_path / "test_store"
        store.save(save_dir)

        # Verify files created inside directory
        assert (save_dir / "index.faiss").exists()
        assert (save_dir / "index.meta.json").exists()
        assert (save_dir / "index.store.json").exists()

        # Load into new store
        store2 = self.VectorStore(backend=mock_backend, index_path=save_dir)

        stats = store2.stats()
        assert stats.total_items == 3

        # Search should work
        results = store2.search("test", k=1)
        assert len(results) == 1

    @pytest.mark.unit
    def test_embedding_cache_deduplication_across_batches(self, mock_backend):
        """Embedding cache prevents duplicate computations across batches."""
        store = self.VectorStore(backend=mock_backend, enable_embedding_cache=True)

        # Create items with duplicate layouts
        layout = LayoutNode(id="same", type=ComponentType.BUTTON, label="Same")
        items = [MockStandardizedData(id=f"item_{i}", layout=layout) for i in range(5)]
        provider = MockProvider(items)

        # Reset embed count
        mock_backend._embed_count = 0

        # Use batch_size=1 so each item is its own batch
        # Cache works ACROSS batches, so first batch embeds, rest hit cache
        store.index_corpus(provider, batch_size=1, workers=1)

        # Should only embed once due to cache (all layouts serialize identically)
        # First item embeds and caches, remaining 4 hit cache
        assert mock_backend._embed_count == 1

    @pytest.mark.unit
    def test_embedding_intra_batch_deduplication(self, mock_backend):
        """Intra-batch deduplication prevents duplicate embeddings within a batch."""
        store = self.VectorStore(backend=mock_backend, enable_embedding_cache=True)

        # Create items with duplicate layouts - all in same batch
        layout = LayoutNode(id="same", type=ComponentType.BUTTON, label="Same")
        items = [MockStandardizedData(id=f"item_{i}", layout=layout) for i in range(5)]
        provider = MockProvider(items)

        # Reset embed count
        mock_backend._embed_count = 0

        # Use large batch_size so all items are in ONE batch
        # Intra-batch deduplication should still only embed once
        store.index_corpus(provider, batch_size=100, workers=1)

        # Should only embed once due to intra-batch deduplication
        assert mock_backend._embed_count == 1

    @pytest.mark.unit
    def test_embedding_cache_disabled(self, mock_backend):
        """Embedding cache disabled means no cross-batch caching."""
        store = self.VectorStore(backend=mock_backend, enable_embedding_cache=False)

        # Create items with UNIQUE layouts (different IDs)
        items = [
            MockStandardizedData(
                id=f"item_{i}",
                layout=LayoutNode(id=f"unique_{i}", type=ComponentType.BUTTON),
            )
            for i in range(5)
        ]
        provider = MockProvider(items)

        mock_backend._embed_count = 0

        # Use batch_size=1 to test cross-batch behavior
        store.index_corpus(provider, batch_size=1, workers=1)

        # Without cache, each batch embeds its own items (no cross-batch reuse)
        # All 5 unique items should be embedded
        assert mock_backend._embed_count == 5

    @pytest.mark.unit
    def test_intra_batch_dedup_with_cache_disabled(self, mock_backend):
        """Intra-batch deduplication works even with cache disabled."""
        store = self.VectorStore(backend=mock_backend, enable_embedding_cache=False)

        # Create items with IDENTICAL layouts in one batch
        layout = LayoutNode(id="same", type=ComponentType.BUTTON, label="Same")
        items = [MockStandardizedData(id=f"item_{i}", layout=layout) for i in range(5)]
        provider = MockProvider(items)

        mock_backend._embed_count = 0

        # All in one batch - intra-batch dedup should still work
        store.index_corpus(provider, batch_size=100, workers=1)

        # Intra-batch dedup: only 1 unique text, so only 1 embed call
        assert mock_backend._embed_count == 1

    @pytest.mark.unit
    def test_clear(self, mock_backend, mock_provider):
        """Clear removes all indexed data."""
        store = self.VectorStore(backend=mock_backend)
        store.index_corpus(mock_provider, batch_size=10, workers=2)

        assert store.stats().total_items == 3

        store.clear()

        assert store.stats().total_items == 0

    @pytest.mark.unit
    def test_search_empty_index(self, mock_backend):
        """Search on empty index returns empty list."""
        store = self.VectorStore(backend=mock_backend)

        results = store.search("anything", k=5)

        assert results == []

    @pytest.mark.unit
    def test_progress_callback(self, mock_backend, mock_provider):
        """Progress callback is invoked during indexing."""
        store = self.VectorStore(backend=mock_backend)

        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        store.index_corpus(
            mock_provider, batch_size=10, workers=2, progress_callback=on_progress
        )

        # Should have received progress updates
        assert len(progress_calls) > 0


@requires_faiss
class TestVectorStorePersistence:
    """Tests for VectorStore persistence edge cases."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock embedding backend."""
        return MockEmbeddingBackend(dimension=64)

    @pytest.fixture(autouse=True)
    def import_vector_store(self):
        """Import VectorStore for tests."""
        from src.vector.lib import VectorStore

        self.VectorStore = VectorStore

    @pytest.mark.unit
    def test_auto_load_existing_index(self, mock_backend, tmp_path):
        """VectorStore auto-loads existing index at path."""
        # Create and save initial store
        store1 = self.VectorStore(backend=mock_backend)
        layout = LayoutNode(id="test", type=ComponentType.BUTTON)
        items = [MockStandardizedData(id="item_0", layout=layout)]
        store1.index_corpus(MockProvider(items), batch_size=10, workers=1)
        save_path = tmp_path / "auto_load"
        store1.save(save_path)

        # Create new store with same path - should auto-load
        store2 = self.VectorStore(backend=mock_backend, index_path=save_path)

        assert store2.stats().total_items == 1

    @pytest.mark.unit
    def test_save_without_path_raises(self, mock_backend):
        """Save without path raises ValueError."""
        store = self.VectorStore(backend=mock_backend)

        with pytest.raises(ValueError, match="No save path"):
            store.save()

    @pytest.mark.unit
    def test_metadata_persists_through_save_load(self, mock_backend, tmp_path):
        """All metadata fields persist through save/load cycle."""
        store = self.VectorStore(backend=mock_backend)

        layout = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            children=[
                LayoutNode(id="child1", type=ComponentType.BUTTON),
                LayoutNode(id="child2", type=ComponentType.INPUT),
            ],
        )
        items = [
            MockStandardizedData(
                id="complex_item", layout=layout, source="custom_source", dataset="v2"
            )
        ]
        store.index_corpus(MockProvider(items), batch_size=10, workers=1)

        # Get original metadata
        original_meta = store.get_metadata("complex_item")

        # Save and load
        save_path = tmp_path / "meta_test"
        store.save(save_path)

        store2 = self.VectorStore(backend=mock_backend, index_path=save_path)
        loaded_meta = store2.get_metadata("complex_item")

        # Verify all fields match
        assert loaded_meta["source"] == original_meta["source"]
        assert loaded_meta["dataset"] == original_meta["dataset"]
        assert loaded_meta["node_count"] == original_meta["node_count"]
        assert loaded_meta["max_depth"] == original_meta["max_depth"]
