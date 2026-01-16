"""Unit tests for the providers module.

Tests for:
- LayoutProvider abstract base class
- Provider registry (register_provider, get_provider, list_providers)
- Provider discovery and instantiation
"""

import pytest

from src.providers import LayoutProvider, get_provider, list_providers


class TestLayoutProviderContract:
    """Tests for LayoutProvider abstract base class contract."""

    @pytest.mark.unit
    def test_layout_provider_is_abstract(self):
        """LayoutProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            LayoutProvider()  # type: ignore

    @pytest.mark.unit
    def test_concrete_provider_requires_name(self):
        """Concrete providers must implement name property."""
        from src.ir import LayoutNode

        class IncompleteProvider(LayoutProvider):
            @property
            def file_extension(self) -> str:
                return ".test"

            def transpile(self, node: LayoutNode) -> str:
                return ""

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    @pytest.mark.unit
    def test_concrete_provider_requires_file_extension(self):
        """Concrete providers must implement file_extension property."""
        from src.ir import LayoutNode

        class IncompleteProvider(LayoutProvider):
            @property
            def name(self) -> str:
                return "incomplete"

            def transpile(self, node: LayoutNode) -> str:
                return ""

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    @pytest.mark.unit
    def test_concrete_provider_requires_transpile(self):
        """Concrete providers must implement transpile method."""

        class IncompleteProvider(LayoutProvider):
            @property
            def name(self) -> str:
                return "incomplete"

            @property
            def file_extension(self) -> str:
                return ".test"

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    @pytest.mark.unit
    def test_transpile_with_context_default_implementation(self):
        """Default transpile_with_context delegates to transpile."""
        from src.ir import LayoutNode, TranspilationContext
        from src.schema import ComponentType

        class TestProvider(LayoutProvider):
            @property
            def name(self) -> str:
                return "test"

            @property
            def file_extension(self) -> str:
                return ".test"

            def transpile(self, node: LayoutNode) -> str:
                return f"transpiled:{node.id}"

        provider = TestProvider()
        node = LayoutNode(id="root", type=ComponentType.CONTAINER)
        context = TranspilationContext(node=node)

        # Both methods should return the same result
        assert provider.transpile(node) == "transpiled:root"
        assert provider.transpile_with_context(context) == "transpiled:root"


class TestProviderRegistry:
    """Tests for provider registry functions."""

    @pytest.mark.unit
    def test_list_providers_returns_list(self):
        """list_providers returns a list."""
        providers = list_providers()
        assert isinstance(providers, list)

    @pytest.mark.unit
    def test_list_providers_includes_d2(self):
        """list_providers includes 'd2' provider."""
        providers = list_providers()
        assert "d2" in providers

    @pytest.mark.unit
    def test_list_providers_includes_plantuml(self):
        """list_providers includes 'plantuml' provider."""
        providers = list_providers()
        assert "plantuml" in providers

    @pytest.mark.unit
    def test_get_provider_d2(self):
        """get_provider returns D2 provider instance."""
        provider = get_provider("d2")
        assert provider.name == "d2"
        assert provider.file_extension == ".d2"
        assert isinstance(provider, LayoutProvider)

    @pytest.mark.unit
    def test_get_provider_plantuml(self):
        """get_provider returns PlantUML provider instance."""
        provider = get_provider("plantuml")
        assert provider.name == "plantuml"
        assert provider.file_extension == ".puml"
        assert isinstance(provider, LayoutProvider)

    @pytest.mark.unit
    def test_get_provider_unknown_raises_key_error(self):
        """get_provider raises KeyError for unknown provider."""
        with pytest.raises(KeyError, match="Unknown provider"):
            get_provider("nonexistent_provider")

    @pytest.mark.unit
    def test_get_provider_error_message_lists_available(self):
        """KeyError message lists available providers."""
        with pytest.raises(KeyError) as exc_info:
            get_provider("nonexistent")

        error_msg = str(exc_info.value)
        assert "d2" in error_msg or "plantuml" in error_msg

    @pytest.mark.unit
    def test_get_provider_returns_new_instance_each_time(self):
        """get_provider returns a new instance on each call."""
        provider1 = get_provider("d2")
        provider2 = get_provider("d2")

        # Should be different instances
        assert provider1 is not provider2
        # But same type
        assert type(provider1) is type(provider2)


class TestProviderTranspilation:
    """Tests for provider transpilation functionality."""

    @pytest.fixture
    def simple_node(self):
        """Create a simple LayoutNode for testing."""
        from src.mid import LayoutNode, Orientation
        from src.schema import ComponentType

        return LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            orientation=Orientation.VERTICAL,
            children=[
                LayoutNode(id="header", type=ComponentType.TOOLBAR),
                LayoutNode(id="content", type=ComponentType.CARD, label="Content"),
            ],
        )

    @pytest.mark.unit
    def test_d2_transpile_produces_valid_dsl(self, simple_node):
        """D2 provider produces valid D2 DSL."""
        provider = get_provider("d2")
        dsl = provider.transpile(simple_node)

        assert dsl is not None
        assert len(dsl) > 0
        # D2 uses curly braces for blocks
        assert "{" in dsl
        assert "}" in dsl
        # Should reference the node id
        assert "root" in dsl.lower() or "container" in dsl.lower()

    @pytest.mark.unit
    def test_plantuml_transpile_produces_valid_dsl(self, simple_node):
        """PlantUML provider produces valid Salt DSL."""
        provider = get_provider("plantuml")
        dsl = provider.transpile(simple_node)

        assert dsl is not None
        assert len(dsl) > 0
        # PlantUML Salt syntax
        assert "@startsalt" in dsl
        assert "@endsalt" in dsl

    @pytest.mark.unit
    def test_d2_transpile_handles_nested_structure(self, simple_node):
        """D2 provider correctly transpiles nested nodes."""
        provider = get_provider("d2")
        dsl = provider.transpile(simple_node)

        # Should have balanced braces (basic DSL validity)
        assert dsl.count("{") == dsl.count("}")

    @pytest.mark.unit
    def test_plantuml_transpile_handles_nested_structure(self, simple_node):
        """PlantUML provider correctly transpiles nested nodes."""
        provider = get_provider("plantuml")
        dsl = provider.transpile(simple_node)

        # Should have balanced braces (basic DSL validity)
        assert dsl.count("{") == dsl.count("}")

    @pytest.mark.unit
    def test_transpile_empty_children(self):
        """Providers handle nodes with no children."""
        from src.mid import LayoutNode
        from src.schema import ComponentType

        node = LayoutNode(id="leaf", type=ComponentType.BUTTON, label="Click")

        for name in ["d2", "plantuml"]:
            provider = get_provider(name)
            dsl = provider.transpile(node)
            assert dsl is not None
            assert len(dsl) > 0


class TestRegisterProviderDecorator:
    """Tests for register_provider decorator."""

    @pytest.mark.unit
    def test_register_provider_adds_to_registry(self):
        """register_provider adds provider to internal registry."""
        from src.ir import LayoutNode
        from src.providers.lib import _registry, register_provider

        # Create a test provider
        @register_provider
        class TempTestProvider(LayoutProvider):
            @property
            def name(self) -> str:
                return "temp_test_provider"

            @property
            def file_extension(self) -> str:
                return ".tmp"

            def transpile(self, node: LayoutNode) -> str:
                return "temp"

        try:
            # Should be in registry
            assert "temp_test_provider" in _registry

            # Should be retrievable
            provider = get_provider("temp_test_provider")
            assert provider.name == "temp_test_provider"
        finally:
            # Cleanup - remove test provider from registry
            _registry.pop("temp_test_provider", None)

    @pytest.mark.unit
    def test_register_provider_returns_class(self):
        """register_provider returns the class for decorator chaining."""
        from src.ir import LayoutNode
        from src.providers.lib import _registry, register_provider

        @register_provider
        class ChainTestProvider(LayoutProvider):
            @property
            def name(self) -> str:
                return "chain_test"

            @property
            def file_extension(self) -> str:
                return ".chain"

            def transpile(self, node: LayoutNode) -> str:
                return "chain"

        try:
            # Decorator should return the class itself
            assert ChainTestProvider is not None
            assert issubclass(ChainTestProvider, LayoutProvider)

            # Class should be instantiable
            instance = ChainTestProvider()
            assert instance.name == "chain_test"
        finally:
            # Cleanup
            _registry.pop("chain_test", None)
