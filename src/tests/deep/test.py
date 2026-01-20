"""Deep testing suite for comprehensive validation.

Converted from deep_testing/ scripts to proper pytest tests.
These tests verify:
- Full pipeline functionality with mock LLM
- Edge cases and stress conditions
- Component type coverage
- Transpiler depth limits
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.llm.conftest import MockLLMBackend
from src.mid import LayoutNode, Orientation
from src.schema import ComponentType

if TYPE_CHECKING:
    from src.render import RenderClient


# =============================================================================
# Full Pipeline Tests (from workbench.py)
# =============================================================================


class TestFullPipelineWithMockLLM:
    """Full pipeline tests using mock LLM (converted from workbench.py)."""

    @pytest.fixture
    def mock_backend(self) -> MockLLMBackend:
        return MockLLMBackend()

    @pytest.mark.unit
    def test_login_generation(self, mock_backend):
        """Generate login layout with mock LLM."""
        from src.llm.generator import GeneratorConfig, LayoutGenerator

        generator = LayoutGenerator(
            backend=mock_backend,
            config=GeneratorConfig(max_retries=1),
        )

        result = generator.generate("login form")
        node = result.context.node

        assert node is not None
        assert node.id == "login-page"
        assert len(node.children) >= 1

    @pytest.mark.unit
    def test_dashboard_generation(self, mock_backend):
        """Generate dashboard layout with mock LLM."""
        from src.llm.generator import GeneratorConfig, LayoutGenerator

        generator = LayoutGenerator(
            backend=mock_backend,
            config=GeneratorConfig(max_retries=1),
        )

        result = generator.generate("dashboard")
        node = result.context.node

        assert node is not None
        assert node.id == "dashboard-root"

    @pytest.mark.unit
    def test_transpile_generated_layout(self, mock_backend):
        """Transpile generated layout to DSL."""
        from src.llm.generator import GeneratorConfig, LayoutGenerator
        from src.providers import get_provider

        generator = LayoutGenerator(
            backend=mock_backend,
            config=GeneratorConfig(max_retries=1),
        )

        result = generator.generate("login form")
        node = result.context.node

        provider = get_provider("plantuml")
        dsl = provider.transpile(node)

        assert dsl is not None
        assert len(dsl) > 0
        assert "@startsalt" in dsl or "{" in dsl

    @pytest.mark.kroki
    @pytest.mark.integration
    def test_full_pipeline_with_render(self, mock_backend, kroki_client: RenderClient):
        """Full pipeline: generate -> transpile -> render."""
        from src.llm.generator import GeneratorConfig, LayoutGenerator
        from src.providers import get_provider

        generator = LayoutGenerator(
            backend=mock_backend,
            config=GeneratorConfig(max_retries=1),
        )

        result = generator.generate("login form")
        node = result.context.node

        provider = get_provider("plantuml")
        dsl = provider.transpile(node)

        # Render with Kroki
        render_result = kroki_client.render(dsl, "plantuml")
        assert render_result.size_bytes > 0


# =============================================================================
# Validation Edge Case Tests (from stress_test.py)
# =============================================================================


class TestValidationEdgeCases:
    """Validation edge case tests (converted from stress_test.py)."""

    @pytest.mark.unit
    def test_invalid_flex_ratio_rejected(self):
        """Invalid flex ratio (>12) is rejected by Pydantic."""
        with pytest.raises(ValueError):
            LayoutNode(
                id="root",
                type=ComponentType.CONTAINER,
                flex_ratio=15,  # Invalid > 12
            )


# =============================================================================
# Transpiler Stress Tests (from transpiler_stress.py)
# =============================================================================


class TestTranspilerDepthLimits:
    """Transpiler depth limit tests (converted from transpiler_stress.py)."""

    @staticmethod
    def create_deep_layout(depth: int) -> LayoutNode:
        """Create a deeply nested layout.

        Args:
            depth: Number of nesting levels to create.

        Returns:
            Root LayoutNode with nested children.
        """
        root = LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            label="Root",
            orientation=Orientation.VERTICAL,
        )
        current = root

        for i in range(depth):
            child = LayoutNode(
                id=f"level-{i + 1}",
                type=ComponentType.CONTAINER,
                label=f"Level {i + 1}",
            )
            current.children.append(child)
            current = child

        return root

    @pytest.mark.unit
    @pytest.mark.parametrize("depth", [10, 20, 50])
    def test_transpile_deep_layout(self, depth: int):
        """Transpiler handles deep nesting without recursion errors."""
        from src.providers import get_provider

        node = self.create_deep_layout(depth)
        provider = get_provider("plantuml")

        dsl = provider.transpile(node)

        assert dsl is not None
        assert len(dsl) > 0
        # Verify brace balance
        assert dsl.count("{") == dsl.count("}")

    @pytest.mark.unit
    @pytest.mark.slow
    def test_transpile_extreme_depth(self):
        """Transpiler handles extreme depth (100 levels)."""
        from src.providers import get_provider

        node = self.create_deep_layout(100)
        provider = get_provider("plantuml")

        # Should not raise RecursionError
        dsl = provider.transpile(node)
        assert dsl is not None

    @pytest.mark.kroki
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("depth", [10, 20])
    def test_render_deep_layout(self, depth: int, kroki_client: RenderClient):
        """Deep layouts render successfully."""
        from src.providers import get_provider
        from src.render import OutputFormat, RenderConfig

        node = self.create_deep_layout(depth)
        provider = get_provider("plantuml")
        dsl = provider.transpile(node)

        # Use SVG for better scaling with large documents
        config = RenderConfig(output_format=OutputFormat.SVG)
        result = kroki_client.render(dsl, "plantuml", config)

        assert result.size_bytes > 0


# =============================================================================
# Component Coverage Tests (from kitchen_sink.py)
# =============================================================================


class TestAllComponentTypes:
    """Test coverage for all component types (converted from kitchen_sink.py)."""

    @pytest.fixture
    def all_component_types(self) -> list[ComponentType]:
        """Get all defined component types."""
        return list(ComponentType)

    @pytest.mark.unit
    def test_all_types_create_valid_nodes(self, all_component_types):
        """All component types can create valid LayoutNodes."""
        for comp_type in all_component_types:
            node = LayoutNode(
                id=f"test-{comp_type.value}",
                type=comp_type,
                label=f"Test {comp_type.value}",
            )
            assert node.id == f"test-{comp_type.value}"
            assert node.type == comp_type.value

    @pytest.mark.unit
    def test_all_types_transpile_plantuml(self, all_component_types):
        """All component types transpile to PlantUML successfully."""
        from src.providers import get_provider

        provider = get_provider("plantuml")

        for comp_type in all_component_types:
            node = LayoutNode(
                id=f"test-{comp_type.value}",
                type=comp_type,
                label=f"Test {comp_type.value}",
            )

            dsl = provider.transpile(node)
            assert dsl is not None, f"Failed to transpile {comp_type}"
            assert len(dsl) > 0

    @pytest.mark.unit
    def test_all_types_transpile_d2(self, all_component_types):
        """All component types transpile to D2 successfully."""
        from src.providers import get_provider

        provider = get_provider("d2")

        for comp_type in all_component_types:
            node = LayoutNode(
                id=f"test-{comp_type.value}",
                type=comp_type,
                label=f"Test {comp_type.value}",
            )

            dsl = provider.transpile(node)
            assert dsl is not None, f"Failed to transpile {comp_type}"
            assert len(dsl) > 0

    @pytest.mark.kroki
    @pytest.mark.integration
    @pytest.mark.slow
    def test_all_types_render(self, all_component_types, kroki_client: RenderClient):
        """All component types render successfully."""
        from src.providers import get_provider

        provider = get_provider("plantuml")
        failed_types = []

        for comp_type in all_component_types:
            node = LayoutNode(
                id=f"test-{comp_type.value}",
                type=comp_type,
                label=f"Test {comp_type.value}",
            )

            dsl = provider.transpile(node)

            try:
                result = kroki_client.render(dsl, "plantuml")
                assert result.size_bytes > 0
            except Exception as e:
                failed_types.append((comp_type, str(e)))

        if failed_types:
            pytest.fail(f"Failed to render types: {failed_types}")
