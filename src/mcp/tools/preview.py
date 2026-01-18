"""Preview layout tool for MCP server.

This tool renders layouts to PNG or SVG wireframe images via Kroki service.
The style parameter abstracts away provider details from the user.
"""

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Style to provider mapping - abstracts implementation details
STYLE_PROVIDERS: dict[str, tuple[str, dict[str, Any]]] = {
    "wireframe": ("plantuml", {}),  # Clean UI mockups (default)
    "sketch": ("d2", {"sketch": True}),  # Hand-drawn appearance
    "minimal": ("d2", {}),  # Simple boxes
}


def preview_layout(
    layout: dict[str, Any],
    style: str = "wireframe",
    format: str = "png",
) -> dict[str, Any]:
    """Render a layout to a visual wireframe image.

    Use this tool to see a visual preview of your layout. The style
    parameter controls the visual appearance without exposing internal
    diagram providers.

    Args:
        layout: Layout JSON to render. Should be the output from
            generate_layout or a manually constructed layout dict.
        style: Visual style for the wireframe. Options:
            - "wireframe": Clean UI mockup (default, best for app/web interfaces)
            - "sketch": Hand-drawn appearance (good for early concepts)
            - "minimal": Simple boxes (good for architecture diagrams)
        format: Output image format. Options:
            - "png": PNG image (default, best for previews)
            - "svg": SVG vector image (scalable, good for docs)

    Returns:
        Dictionary containing:
        - image_data: Base64-encoded image data
        - format: Image format used
        - style: Visual style used
        - size_bytes: Image size in bytes

    Raises:
        ValueError: If style or format is invalid.
        RuntimeError: If Kroki service is unavailable or rendering fails.

    Example:
        >>> result = preview_layout(layout, style="wireframe")
        >>> # Decode and save:
        >>> import base64
        >>> with open("preview.png", "wb") as f:
        ...     f.write(base64.b64decode(result["image_data"]))
    """
    from pydantic import ValidationError as PydanticValidationError

    from src.mid import LayoutNode
    from src.render import OutputFormat, RenderClient, RenderConfig, RenderError

    # Validate style
    if style not in STYLE_PROVIDERS:
        valid_styles = sorted(STYLE_PROVIDERS.keys())
        raise ValueError(f"Invalid style: {style}. Options: {', '.join(valid_styles)}")

    # Get provider configuration from style
    provider, provider_options = STYLE_PROVIDERS[style]

    # Validate format
    try:
        output_format = OutputFormat(format.lower())
    except ValueError as e:
        raise ValueError(
            f"Invalid format: {format}. Options: png, svg, pdf, jpeg"
        ) from e

    # Parse layout
    try:
        node = LayoutNode.model_validate(layout)
    except PydanticValidationError as e:
        raise ValueError(f"Invalid layout structure: {e}") from e

    # Create render client
    client = RenderClient()

    # Check availability
    if not client.is_available():
        raise RuntimeError(
            "Kroki service is not available. Start it with: python . docker up kroki"
        )

    # Configure rendering
    config = RenderConfig(
        output_format=output_format,
    )

    try:
        # Render using internal provider
        logger.info(f"Rendering preview with style={style} ({provider}) to {format}")
        result = client.render_layout(node, provider=provider, config=config)

        # Encode image as base64
        image_b64 = base64.b64encode(result.image_bytes).decode("ascii")

        return {
            "image_data": image_b64,
            "format": format,
            "style": style,
            "size_bytes": result.size_bytes,
        }

    except RenderError as e:
        logger.error(f"Rendering failed: {e}")
        raise RuntimeError(f"Rendering failed: {e}") from e


__all__ = ["preview_layout", "STYLE_PROVIDERS"]
