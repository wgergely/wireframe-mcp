"""A/B benchmark comparing RAG vs non-RAG generation."""

from dataclasses import dataclass

import pytest


@dataclass
class BenchmarkResult:
    """Results from a single generation."""

    query: str
    rag_enabled: bool
    node_count: int
    max_depth: int
    component_types: set[str]
    generation_time_ms: float


BENCHMARK_QUERIES = [
    "login form with email and password",
    "dashboard with sidebar and stats",
    "settings page with toggle switches",
    "e-commerce product card",
    "navigation bar with menu items",
]


@pytest.mark.slow
@pytest.mark.rag
class TestRAGABBenchmark:
    """A/B benchmark for RAG vs non-RAG generation quality."""

    @pytest.fixture
    def mock_llm_backend(self):
        """Mock LLM backend for testing without API calls."""
        from unittest.mock import MagicMock

        backend = MagicMock()

        # Create sample layout JSON
        sample_layout = {
            "id": "root",
            "type": "container",
            "children": [
                {"id": "child1", "type": "button", "children": []},
                {"id": "child2", "type": "input", "children": []},
            ],
        }

        backend.generate.return_value = MagicMock(
            content=str(sample_layout).replace("'", '"'),
            model="mock-model",
            usage={"total_tokens": 100},
        )
        return backend

    def test_rag_vs_no_rag_component_coverage(self):
        """RAG should improve component type coverage."""
        # This test documents that we WANT to measure RAG impact
        # but cannot run without actual LLM calls
        #
        # When RAG index and LLM are available:
        # 1. Generate layouts with RAG enabled
        # 2. Generate layouts with RAG disabled
        # 3. Compare component variety, depth, structural quality
        pytest.skip("Requires LLM and RAG index - run manually in full environment")

    def _analyze(self, query: str, rag: bool, output) -> BenchmarkResult:
        """Analyze generation output.

        Args:
            query: The original query string.
            rag: Whether RAG was enabled.
            output: Generation output with context.node attribute.

        Returns:
            BenchmarkResult with metrics.
        """
        from src.mid import LayoutNode

        node: LayoutNode = output.context.node

        def count_nodes(n: LayoutNode, depth: int = 0) -> tuple[int, int]:
            count = 1
            max_d = depth
            for child in n.children:
                c, d = count_nodes(child, depth + 1)
                count += c
                max_d = max(max_d, d)
            return count, max_d

        def get_types(n: LayoutNode) -> set[str]:
            # Handle both enum and string values (use_enum_values=True)
            node_type = n.type.value if hasattr(n.type, "value") else n.type
            types = {node_type}
            for child in n.children:
                types.update(get_types(child))
            return types

        count, max_depth = count_nodes(node)
        types = get_types(node)

        return BenchmarkResult(
            query=query,
            rag_enabled=rag,
            node_count=count,
            max_depth=max_depth,
            component_types=types,
            generation_time_ms=0,
        )


class TestBenchmarkResultDataclass:
    """Unit tests for BenchmarkResult dataclass."""

    @pytest.mark.unit
    def test_benchmark_result_creation(self):
        """BenchmarkResult can be created with all fields."""
        result = BenchmarkResult(
            query="test query",
            rag_enabled=True,
            node_count=5,
            max_depth=3,
            component_types={"container", "button"},
            generation_time_ms=150.5,
        )

        assert result.query == "test query"
        assert result.rag_enabled is True
        assert result.node_count == 5
        assert result.max_depth == 3
        assert result.component_types == {"container", "button"}
        assert result.generation_time_ms == 150.5

    @pytest.mark.unit
    def test_benchmark_queries_defined(self):
        """Benchmark queries are defined."""
        assert len(BENCHMARK_QUERIES) >= 5
        assert all(isinstance(q, str) for q in BENCHMARK_QUERIES)
