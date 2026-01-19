"""Provider abstraction for DSL transpilation.

This module defines the abstract base class for layout providers and
provides a registry/factory for accessing them by name.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from src.ir import LayoutNode, TranspilationContext


class LayoutFeature(str, Enum):
    """Standard layout features that providers may or may not support.

    Used to declare provider capabilities and emit warnings when
    a feature is used but not supported by the target provider.
    """

    # Layout modes
    DISPLAY_FLEX = "display_flex"
    DISPLAY_GRID = "display_grid"
    DISPLAY_BLOCK = "display_block"

    # Orientation
    ORIENTATION_HORIZONTAL = "orientation_horizontal"
    ORIENTATION_VERTICAL = "orientation_vertical"
    ORIENTATION_OVERLAY = "orientation_overlay"

    # Sizing
    FLEX_RATIO = "flex_ratio"
    FIXED_WIDTH = "fixed_width"
    FIXED_HEIGHT = "fixed_height"
    GRID_COLUMNS = "grid_columns"
    GRID_ROWS = "grid_rows"

    # Spacing
    GAP = "gap"
    PADDING = "padding"

    # Alignment
    ALIGN = "align"
    JUSTIFY = "justify"
    ALIGN_CONTENT = "align_content"
    ALIGN_SELF = "align_self"
    WRAP = "wrap"

    # Text styling
    TEXT_SIZE = "text_size"
    TEXT_WEIGHT = "text_weight"
    TEXT_TRANSFORM = "text_transform"
    TEXT_ALIGN = "text_align"
    SEMANTIC_COLOR = "semantic_color"

    # Container features
    SCROLLABLE = "scrollable"


@dataclass
class TranspilationWarning:
    """Warning emitted when a feature cannot be fully represented.

    Attributes:
        feature: The LayoutFeature that triggered the warning.
        node_id: ID of the node where the issue occurred.
        message: Human-readable explanation.
        value: The value that couldn't be represented (optional).
    """

    feature: LayoutFeature
    node_id: str
    message: str
    value: str | None = None


@dataclass
class TranspilationResult:
    """Result of transpilation including DSL code and any warnings.

    Attributes:
        dsl_code: The generated DSL code.
        warnings: List of warnings about unsupported features.
        provider: Name of the provider that generated this result.
    """

    dsl_code: str
    warnings: list[TranspilationWarning] = field(default_factory=list)
    provider: str = ""

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were emitted."""
        return len(self.warnings) > 0


class LayoutProvider(ABC):
    """Abstract base class for layout DSL providers.

    Each provider implements transpilation from LayoutNode IR to a
    specific DSL syntax (e.g., D2, PlantUML Salt).

    Subclasses must implement:
        - name: Provider identifier string
        - file_extension: Output file extension
        - supported_formats: Set of output formats (png, svg, pdf, jpeg)
        - transpile: IR to DSL conversion

    Optionally implement transpile_with_context for RAG-aware transpilation.

    Example:
        >>> class MyProvider(LayoutProvider):
        ...     name = "my_dsl"
        ...     file_extension = ".dsl"
        ...     supported_formats = frozenset({"svg", "png"})
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

    @property
    @abstractmethod
    def supported_formats(self) -> frozenset[str]:
        """Supported output formats for rendering via Kroki.

        Returns:
            frozenset of format strings (e.g., {"svg", "png", "pdf"}).
        """
        ...

    @property
    @abstractmethod
    def supported_features(self) -> frozenset[LayoutFeature]:
        """Layout features this provider can represent.

        Used for capability checking and warning generation.

        Returns:
            frozenset of LayoutFeature values this provider supports.
        """
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

    def transpile_with_context(self, context: TranspilationContext) -> str:
        """Transpile with full context including RAG results.

        Providers can override this to use similar_layouts, hints,
        and metadata from the context. Default implementation
        delegates to transpile().

        Args:
            context: Full transpilation context with RAG data.

        Returns:
            str: DSL code representing the layout.
        """
        return self.transpile(context.node)

    def transpile_with_warnings(self, node: LayoutNode) -> TranspilationResult:
        """Transpile and collect warnings for unsupported features.

        This method transpiles the node and also checks which features
        in the layout tree cannot be represented by this provider.

        Args:
            node: The root LayoutNode to transpile.

        Returns:
            TranspilationResult with DSL code and warnings.
        """
        dsl_code = self.transpile(node)
        warnings = self._collect_warnings(node)
        return TranspilationResult(
            dsl_code=dsl_code,
            warnings=warnings,
            provider=self.name,
        )

    def _collect_warnings(self, node: LayoutNode) -> list[TranspilationWarning]:
        """Recursively collect warnings for unsupported features.

        Args:
            node: Node to check.

        Returns:
            List of warnings for this node and all descendants.
        """
        warnings: list[TranspilationWarning] = []
        supported = self.supported_features

        # Check each feature used by this node
        warnings.extend(self._check_node_features(node, supported))

        # Recurse into children
        for child in node.children:
            warnings.extend(self._collect_warnings(child))

        return warnings

    def _check_node_features(
        self, node: LayoutNode, supported: frozenset[LayoutFeature]
    ) -> list[TranspilationWarning]:
        """Check which features a node uses that aren't supported.

        Args:
            node: Node to check.
            supported: Set of supported features.

        Returns:
            List of warnings for unsupported features.
        """
        warnings: list[TranspilationWarning] = []

        # Get node values (handle Pydantic use_enum_values)
        orientation = node.orientation
        if isinstance(orientation, str):
            orientation = orientation

        display = getattr(node, "display", "flex")
        if hasattr(display, "value"):
            display = display.value

        # Check display mode
        if display == "grid" and LayoutFeature.DISPLAY_GRID not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.DISPLAY_GRID,
                    node_id=node.id,
                    message=f"Grid display mode not supported by {self.name}",
                    value="grid",
                )
            )

        # Check orientation
        if (
            orientation == "overlay"
            and LayoutFeature.ORIENTATION_OVERLAY not in supported
        ):
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.ORIENTATION_OVERLAY,
                    node_id=node.id,
                    message=f"Overlay orientation not supported by {self.name}",
                    value="overlay",
                )
            )

        # Check flex_ratio (only warn if non-default and unsupported)
        if node.flex_ratio != 1 and LayoutFeature.FLEX_RATIO not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.FLEX_RATIO,
                    node_id=node.id,
                    message=f"flex_ratio={node.flex_ratio} ignored by {self.name}",
                    value=str(node.flex_ratio),
                )
            )

        # Check grid properties
        grid_cols = getattr(node, "grid_columns", None)
        grid_rows = getattr(node, "grid_rows", None)
        if grid_cols and LayoutFeature.GRID_COLUMNS not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.GRID_COLUMNS,
                    node_id=node.id,
                    message=f"grid_columns={grid_cols} ignored by {self.name}",
                    value=str(grid_cols),
                )
            )
        if grid_rows and LayoutFeature.GRID_ROWS not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.GRID_ROWS,
                    node_id=node.id,
                    message=f"grid_rows={grid_rows} ignored by {self.name}",
                    value=str(grid_rows),
                )
            )

        # Check alignment properties
        if node.justify and LayoutFeature.JUSTIFY not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.JUSTIFY,
                    node_id=node.id,
                    message=f"justify={node.justify} ignored by {self.name}",
                    value=str(node.justify),
                )
            )
        if node.align and LayoutFeature.ALIGN not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.ALIGN,
                    node_id=node.id,
                    message=f"align={node.align} ignored by {self.name}",
                    value=str(node.align),
                )
            )
        if node.align_content and LayoutFeature.ALIGN_CONTENT not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.ALIGN_CONTENT,
                    node_id=node.id,
                    message=f"align_content ignored by {self.name}",
                    value=str(node.align_content),
                )
            )
        if node.align_self and LayoutFeature.ALIGN_SELF not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.ALIGN_SELF,
                    node_id=node.id,
                    message=f"align_self={node.align_self} ignored by {self.name}",
                    value=str(node.align_self),
                )
            )
        if node.wrap and LayoutFeature.WRAP not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.WRAP,
                    node_id=node.id,
                    message=f"wrap={node.wrap} ignored by {self.name}",
                    value=str(node.wrap),
                )
            )

        # Check scrollable (only if true and unsupported)
        if node.scrollable and LayoutFeature.SCROLLABLE not in supported:
            warnings.append(
                TranspilationWarning(
                    feature=LayoutFeature.SCROLLABLE,
                    node_id=node.id,
                    message=f"scrollable=true ignored by {self.name}",
                    value="true",
                )
            )

        return warnings


# Provider registry - populated by provider modules on import
_registry: dict[str, type[LayoutProvider]] = {}


def register_provider(provider_cls: type[LayoutProvider]) -> type[LayoutProvider]:
    """Register a provider class in the registry.

    Uses a temporary instance to retrieve the provider name.

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
    # Instantiate once to get the name property
    _registry[provider_cls().name] = provider_cls
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


def get_provider_formats(name: str) -> frozenset[str]:
    """Get supported output formats for a provider.

    Args:
        name: The provider identifier (e.g., "d2", "plantuml").

    Returns:
        frozenset[str]: Set of supported format strings.

    Raises:
        KeyError: If no provider with the given name is registered.

    Example:
        >>> get_provider_formats("plantuml")
        frozenset({'png', 'svg', 'pdf', 'jpeg'})
        >>> get_provider_formats("d2")
        frozenset({'svg'})
    """
    provider = get_provider(name)
    return provider.supported_formats


def _import_providers() -> None:
    """Import provider modules to trigger registration."""
    import importlib

    for module_name in ("d2", "plantuml"):
        try:
            importlib.import_module(f"src.providers.{module_name}")
        except ImportError:
            pass
