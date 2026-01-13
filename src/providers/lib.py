"""Provider abstraction for DSL transpilation.

This module defines the abstract base class for layout providers and
provides a registry/factory for accessing them by name.
"""

from abc import ABC, abstractmethod

from src.ir import LayoutNode


class LayoutProvider(ABC):
    """Abstract base class for layout DSL providers.

    Each provider implements transpilation from LayoutNode IR to a
    specific DSL syntax (e.g., D2, PlantUML Salt).

    Subclasses must implement:
        - name: Provider identifier string
        - file_extension: Output file extension
        - transpile: IR to DSL conversion

    Example:
        >>> class MyProvider(LayoutProvider):
        ...     name = "my_dsl"
        ...     file_extension = ".dsl"
        ...     def transpile(self, node: LayoutNode) -> str:
        ...         return f"node: {node.id}"
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string."""
        ...

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Output file extension (e.g., '.d2', '.puml')."""
        ...

    @abstractmethod
    def transpile(self, node: LayoutNode) -> str:
        """Transpile a LayoutNode to DSL syntax.

        Args:
            node: The root LayoutNode to transpile.

        Returns:
            str: DSL code representing the layout.
        """
        ...


# Provider registry - populated by provider modules on import
_registry: dict[str, type[LayoutProvider]] = {}


def register_provider(provider_cls: type[LayoutProvider]) -> type[LayoutProvider]:
    """Register a provider class in the registry.

    This is typically used as a decorator on provider classes.

    Args:
        provider_cls: The provider class to register.

    Returns:
        The provider class (for decorator chaining).

    Example:
        >>> @register_provider
        ... class MyProvider(LayoutProvider):
        ...     name = "my_dsl"
        ...     ...
    """
    # Instantiate to get the name property
    instance = provider_cls()
    _registry[instance.name] = provider_cls
    return provider_cls


def get_provider(name: str) -> LayoutProvider:
    """Get a provider instance by name.

    Args:
        name: The provider identifier (e.g., "d2", "plantuml").

    Returns:
        LayoutProvider: An instance of the requested provider.

    Raises:
        KeyError: If no provider with the given name is registered.

    Example:
        >>> provider = get_provider("d2")
        >>> provider.transpile(node)
    """
    if name not in _registry:
        # Attempt to import provider modules to trigger registration
        _import_providers()
        if name not in _registry:
            available = ", ".join(_registry.keys()) or "(none)"
            raise KeyError(f"Unknown provider '{name}'. Available: {available}")
    return _registry[name]()


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        list[str]: List of provider identifier strings.

    Example:
        >>> list_providers()
        ['d2', 'plantuml']
    """
    _import_providers()
    return list(_registry.keys())


def _import_providers() -> None:
    """Import provider modules to trigger registration."""
    import importlib

    for module_name in ("d2", "plantuml"):
        try:
            importlib.import_module(f"src.providers.{module_name}")
        except ImportError:
            pass
