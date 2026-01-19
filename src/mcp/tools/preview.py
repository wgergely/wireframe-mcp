"""Preview layout tool for MCP server.

Renders layouts to wireframe images via Kroki service.
Provider configuration is controlled via environment variables.

Supported formats vary by provider:
- plantuml: png, svg, pdf, jpeg
- d2: svg only (Kroki limitation)
"""

import base64
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_render_client():
    """Get cached RenderClient instance."""
    from src.render import RenderClient

    return RenderClient()


def _get_preview_config() -> tuple[str, bool]:
    """Get preview provider configuration from environment.

    Returns:
        Tuple of (provider, sketch_mode).
    """
    from src.config import EnvVar, get_environment

    provider = get_environment(EnvVar.MCP_PREVIEW_PROVIDER)
    sketch = get_environment(EnvVar.MCP_PREVIEW_SKETCH)
    return provider, sketch


def _validate_format_for_provider(output_format: str, provider: str) -> None:
    """Validate output format against provider's supported formats.

    Args:
        output_format: Requested output format.
        provider: Provider name to validate against.

    Raises:
        ValueError: If format not supported by the provider.
    """
    from src.providers import get_provider_formats

    supported = get_provider_formats(provider)
    if output_format.lower() not in supported:
        raise ValueError(
            f"Format '{output_format}' not supported by provider '{provider}'. "
            f"Supported: {sorted(supported)}"
        )


def preview_layout(
    layout: dict[str, Any],
    output_format: str = "svg",
) -> dict[str, Any]:
    """Render a layout to a wireframe image.

    Converts the layout JSON to a visual wireframe via Kroki service.
    The rendering provider is configured via environment variables:
    - MCP_PREVIEW_PROVIDER: "plantuml" (default) or "d2"
    - MCP_PREVIEW_SKETCH: Enable D2 hand-drawn mode (default: false)

    Note: Available formats depend on the configured provider:
    - plantuml: png, svg, pdf, jpeg
    - d2: svg only (Kroki limitation)

    Args:
        layout: Layout JSON to render. Should be the output from
            generate_layout or a manually constructed layout dict.
        output_format: Output format. Default: "svg" (universal support)

    Returns:
        Dictionary containing:
        - image_data: Base64-encoded image data
        - format: Image format used
        - provider: Rendering provider used
        - size_bytes: Image size in bytes

    Raises:
        ValueError: If layout structure is invalid or format unsupported.
        RuntimeError: If Kroki service is unavailable or rendering fails.

    Example:
        >>> result = preview_layout(layout)
        >>> # Decode and save:
        >>> import base64
        >>> with open("preview.svg", "wb") as f:
        ...     f.write(base64.b64decode(result["image_data"]))
    """
    from pydantic import ValidationError as PydanticValidationError

    from src.mid import LayoutNode
    from src.render import OutputFormat, RenderConfig, RenderError

    # Get provider configuration from environment
    provider, sketch_mode = _get_preview_config()

    # Validate format against provider capabilities
    _validate_format_for_provider(output_format, provider)

    # Parse format enum
    try:
        fmt = OutputFormat(output_format.lower())
    except ValueError as e:
        valid_formats = [f.value for f in OutputFormat]
        raise ValueError(
            f"Invalid format: {output_format}. Options: {', '.join(valid_formats)}"
        ) from e

    # Parse layout
    try:
        node = LayoutNode.model_validate(layout)
    except PydanticValidationError as e:
        raise ValueError(f"Invalid layout structure: {e}") from e

    # Get cached render client
    client = _get_render_client()

    # Check availability
    if not client.is_available():
        raise RuntimeError(
            "Kroki service is not available. Start it with: python . service start"
        )

    # Configure rendering
    config = RenderConfig(
        output_format=fmt,
        sketch=sketch_mode if provider == "d2" else False,
    )

    try:
        logger.info(f"Rendering: provider={provider}, format={output_format}")
        result = client.render_layout(node, provider=provider, config=config)

        # Encode image as base64
        image_b64 = base64.b64encode(result.image_bytes).decode("ascii")

        return {
            "image_data": image_b64,
            "format": output_format,
            "provider": provider,
            "size_bytes": result.size_bytes,
        }

    except RenderError as e:
        logger.error(f"Rendering failed: {e}")
        raise RuntimeError(f"Rendering failed: {e}") from e


__all__ = ["preview_layout"]
