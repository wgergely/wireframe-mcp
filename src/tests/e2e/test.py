"""End-to-end integration tests for full pipeline.

Tests the complete flow: NL Query → LLM → IR → Transpile → Output
"""

import pytest

from src.ir import TranspilationContext
from src.llm import LayoutGenerator
from src.output import OutputGenerator
from src.providers import get_provider

# =============================================================================
# Parametrized E2E Test Cases
# =============================================================================

# Each test case: (query, min_nodes, max_depth, expected_types)
E2E_TEST_CASES = [
    pytest.param(
        "empty container box",
        1,  # min nodes
        1,  # expected max depth
        ["container"],  # at least these types
        id="minimal-simple-box",
    ),
    pytest.param(
        "login form with email and password input fields",
        3,  # min nodes (container + 2 inputs)
        2,  # expected max depth
        ["input"],
        id="low-complexity-form",
    ),
    pytest.param(
        "dashboard with sidebar navigation and main content area",
        4,  # min nodes
        2,
        ["container"],
        id="medium-complexity-dashboard",
    ),
    pytest.param(
        "complex app with toolbar header, sidebar with nested menu items, "
        "main content area with card grid containing multiple cards each with "
        "title, description, and action buttons",
        10,  # min nodes for deeply nested
        4,  # expect deeper nesting
        ["container", "toolbar", "card", "button"],
        id="high-complexity-nested-app",
    ),
    pytest.param(
        "settings panel with three tabs: general settings with toggle switches, "
        "notifications with checkbox options, and account with profile card",
        8,
        3,
        ["container"],
        id="high-complexity-tabbed-settings",
    ),
]


def count_nodes(node) -> int:
    """Count total nodes in tree."""
    return 1 + sum(count_nodes(child) for child in node.children)


def max_depth(node, current=1) -> int:
    """Calculate maximum depth of tree."""
    if not node.children:
        return current
    return max(max_depth(child, current + 1) for child in node.children)


def collect_types(node, types=None) -> set:
    """Collect all component types in tree."""
    if types is None:
        types = set()
    type_val = node.type if isinstance(node.type, str) else node.type.value
    types.add(type_val)
    for child in node.children:
        collect_types(child, types)
    return types


class TestFullPipelineParametrized:
    """Parametrized E2E tests covering query complexity spectrum."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query,min_nodes,expected_depth,expected_types",
        E2E_TEST_CASES,
    )
    def test_query_complexity_spectrum(
        self, query, min_nodes, expected_depth, expected_types
    ):
        """Test full pipeline with varied complexity queries.

        Validates:
        1. Generation succeeds
        2. Node count meets minimum for complexity
        3. Tree depth matches expected complexity
        4. Expected component types are present
        5. Output formats work (tree + DSL)
        """
        # Generate layout
        generator = LayoutGenerator()
        result = generator.generate(query)

        # Verify context
        assert isinstance(result.context, TranspilationContext)
        assert result.context.node is not None
        assert result.stats.total_tokens > 0

        node = result.context.node
        node_count = count_nodes(node)
        depth = max_depth(node)
        types = collect_types(node)

        # Validate complexity expectations
        assert node_count >= min_nodes, (
            f"Expected at least {min_nodes} nodes, got {node_count}"
        )
        assert depth >= expected_depth, (
            f"Expected depth >= {expected_depth}, got {depth}"
        )
        for expected_type in expected_types:
            assert expected_type in types, (
                f"Expected type '{expected_type}' not found. Got: {types}"
            )

        # Verify output generation works
        output_gen = OutputGenerator()
        output = output_gen.generate_from_context(result.context)
        assert output.text_tree
        assert output.dsl_code

        # Log results for debugging
        print(f"\n--- Query: {query[:50]}... ---")
        print(f"Nodes: {node_count}, Depth: {depth}, Types: {types}")
        print(f"Tokens: {result.stats.total_tokens}")

    @pytest.mark.integration
    def test_d2_provider_transpilation(self):
        """Test D2 transpilation from generated layout."""
        generator = LayoutGenerator()
        result = generator.generate("card with title and button")

        provider = get_provider("d2")
        dsl = provider.transpile(result.context.node)

        assert dsl
        assert result.context.node.id in dsl
        assert "{" in dsl  # D2 block syntax

    @pytest.mark.integration
    def test_plantuml_provider_transpilation(self):
        """Test PlantUML transpilation from generated layout."""
        generator = LayoutGenerator()
        result = generator.generate("simple toolbar")

        provider = get_provider("plantuml")
        dsl = provider.transpile(result.context.node)

        assert dsl
        # PlantUML Salt syntax
        assert "@startsalt" in dsl or "{" in dsl


# =============================================================================
# RAG-Enhanced E2E Tests (require vector index)
# =============================================================================

# RAG test cases: queries that benefit from similar layout examples
RAG_TEST_CASES = [
    pytest.param(
        "login form with social authentication buttons",
        ["container", "button"],
        id="rag-login-form",
    ),
    pytest.param(
        "dashboard with data visualization cards",
        ["container", "card"],
        id="rag-dashboard",
    ),
    pytest.param(
        "mobile navigation with bottom tab bar",
        ["container"],
        id="rag-mobile-nav",
    ),
]


class TestRAGEnhancedPipeline:
    """E2E tests requiring RAG context from vector store.

    These tests verify the full RAG flow:
    1. Auto-builds index if needed (via fixture)
    2. Retrieves similar layouts from VectorStore
    3. Includes examples in prompt
    4. Validates prompt_context shows RAG was used
    """

    @pytest.mark.integration
    @pytest.mark.rag
    def test_rag_index_loads(self, vector_store):
        """Test that RAG index is available and has items."""
        stats = vector_store.stats()
        assert stats.total_items > 0, "RAG index should have items"
        print(f"RAG index loaded: {stats.total_items} items")

    @pytest.mark.integration
    @pytest.mark.rag
    def test_vector_search_returns_results(self, vector_store):
        """Test that vector search finds similar layouts."""
        results = vector_store.search("login form", k=3)
        assert len(results) > 0, "Should find similar layouts"
        print(f"Found {len(results)} similar layouts for 'login form'")

    @pytest.mark.integration
    @pytest.mark.rag
    @pytest.mark.parametrize("query,expected_types", RAG_TEST_CASES)
    def test_rag_enhanced_generation(self, vector_store, query, expected_types):
        """Test full RAG-enhanced generation pipeline.

        Validates:
        1. VectorStore is used for retrieval
        2. Examples are included in prompt
        3. Generated layout is valid
        """
        from src.llm import GeneratorConfig

        # Create generator WITH vector store
        generator = LayoutGenerator(
            vector_store=vector_store,
            config=GeneratorConfig(
                include_rag_context=True,
                max_rag_examples=3,
            ),
        )

        result = generator.generate(query)

        # CRITICAL: Verify RAG was used
        rag_count = result.prompt_context.example_count
        assert rag_count > 0, f"RAG examples should be included. Got {rag_count}"
        assert len(result.prompt_context.example_ids) > 0, (
            "Should have example IDs from RAG retrieval"
        )

        # Verify layout is valid
        assert result.context.node is not None
        types = collect_types(result.context.node)
        for expected_type in expected_types:
            assert expected_type in types, (
                f"Expected type '{expected_type}' not found. Got: {types}"
            )

        # Log results
        print(f"\n--- RAG Query: {query[:40]}... ---")
        print(f"RAG examples used: {result.prompt_context.example_count}")
        print(f"Example IDs: {result.prompt_context.example_ids}")
        print(f"Types generated: {types}")

    @pytest.mark.integration
    @pytest.mark.rag
    def test_no_rag_context_when_disabled(self, vector_store):
        """Test that RAG can be explicitly disabled."""
        from src.llm import GeneratorConfig

        generator = LayoutGenerator(
            vector_store=vector_store,
            config=GeneratorConfig(include_rag_context=False),
        )

        result = generator.generate("simple box")

        # RAG should be disabled
        assert result.prompt_context.example_count == 0, (
            "RAG should be disabled when include_rag_context=False"
        )
