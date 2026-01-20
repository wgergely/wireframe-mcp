"""Unit tests for MCP tools."""

import pytest


class TestRefineLayout:
    """Tests for refine_layout tool."""

    @pytest.mark.unit
    def test_refine_layout_exists(self):
        """refine_layout function is importable."""
        from src.mcp.tools.refine import refine_layout

        assert callable(refine_layout)

    @pytest.mark.unit
    def test_refine_layout_requires_artifact_id(self):
        """refine_layout requires artifact_id parameter."""
        import inspect

        from src.mcp.tools.refine import refine_layout

        sig = inspect.signature(refine_layout)
        assert "artifact_id" in sig.parameters
        assert "feedback" in sig.parameters


class TestGenerateLayoutStats:
    """Tests for generate_layout stats output."""

    @pytest.mark.unit
    def test_stats_rag_fields_exist(self, mock_generator):
        """Stats includes correct RAG field names."""
        from src.mcp.tools.generate import generate_layout

        # This should NOT crash - currently crashes on prompt_context.examples
        result = generate_layout(
            query="simple login form",
            include_rag=False,  # Simpler path
            persist=False,
        )

        assert "stats" in result
        assert "rag_used" in result["stats"]
        assert "rag_examples" in result["stats"]
        assert isinstance(result["stats"]["rag_used"], bool)
        assert isinstance(result["stats"]["rag_examples"], int)


@pytest.fixture
def mock_generator(monkeypatch):
    """Mock LayoutGenerator class to avoid LLM calls."""
    from src.ir import TranspilationContext
    from src.mid import LayoutNode
    from src.prompt import PromptContext
    from src.schema import ComponentType

    mock_node = LayoutNode(id="root", type=ComponentType.CONTAINER)

    class MockStats:
        attempts = 1
        validation_retries = 0
        json_repairs = 0
        total_tokens = 100
        final_model = "mock-model"

    class MockOutput:
        context = TranspilationContext(node=mock_node)
        prompt_context = PromptContext(
            query="simple login form",
            example_count=0,
            example_ids=[],
        )
        stats = MockStats()
        raw_response = "{}"

    class MockLayoutGenerator:
        def __init__(self, **kwargs):
            pass

        def generate(self, query, **kwargs):
            return MockOutput()

    monkeypatch.setattr("src.llm.LayoutGenerator", MockLayoutGenerator)


class TestGenerateLayoutRAGStatus:
    """Tests for RAG status in generate_layout response."""

    @pytest.mark.unit
    def test_response_includes_rag_status(self, mock_generator):
        """Response includes explicit RAG availability status."""
        from src.mcp.tools.generate import generate_layout

        result = generate_layout(
            query="test",
            include_rag=True,
            persist=False,
        )

        # Should explicitly report RAG status
        assert "rag_status" in result
        assert "available" in result["rag_status"]
        assert "reason" in result["rag_status"]

    @pytest.mark.unit
    def test_rag_status_reports_unavailable_reason(self, mock_generator, monkeypatch):
        """RAG status explains why RAG is unavailable."""
        # Mock get_vector_store to return None (at cache module level)
        monkeypatch.setattr(
            "src.mcp.tools.cache.get_vector_store",
            lambda: None,
        )

        from src.mcp.tools.generate import generate_layout

        result = generate_layout(
            query="test",
            include_rag=True,
            persist=False,
        )

        assert result["rag_status"]["available"] is False
        assert "index" in result["rag_status"]["reason"].lower()
