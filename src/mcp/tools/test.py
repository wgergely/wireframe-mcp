"""Unit tests for MCP tools."""

import pytest


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
