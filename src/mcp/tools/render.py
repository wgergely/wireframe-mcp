"""Render layout tool for MCP server.

This tool renders layouts to PNG or SVG images via Kroki service.
"""

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


def render_layout(
    layout: dict[str, Any],
    format: str = "png",
    provider: str = "plantuml",
) -> dict[str, Any]:
    """Render a layout to an image.

    Converts the layout to DSL code and renders it using the Kroki
    diagram service. Requires Kroki to be running (see docker compose).

    Args:
        layout: Layout JSON to render. Should be the output from
            generate_layout or a manually constructed layout dict.
        format: Output image format. Options:
            - "png": PNG image (default, best for previews)
            - "svg": SVG vector image (scalable, good for docs)
        provider: DSL provider for rendering. Options:
            - "plantuml": PlantUML Salt notation (default, better wireframes)
            - "d2": D2 diagram language (better for architecture)

    Returns:
        Dictionary containing:
        - image_data: Base64-encoded image data
        - format: Image format used
        - size_bytes: Image size in bytes
        - provider: DSL provider used

    Raises:
        ToolError: If Kroki service is unavailable or rendering fails.

    Example:
        >>> result = render_layout(layout, format="png")
        >>> # Decode and save:
        >>> import base64
        >>> with open("output.png", "wb") as f:
        ...     f.write(base64.b64decode(result["image_data"]))
    """
    from pydantic import ValidationError as PydanticValidationError

    from src.mid import LayoutNode
    from src.render import OutputFormat, RenderClient, RenderConfig, RenderError

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
        # Render
        logger.info(f"Rendering layout with {provider} to {format}")
        result = client.render_layout(node, provider=provider, config=config)

        # Encode image as base64
        image_b64 = base64.b64encode(result.image_bytes).decode("ascii")

        return {
            "image_data": image_b64,
            "format": format,
            "size_bytes": result.size_bytes,
            "provider": provider,
        }

    except RenderError as e:
        logger.error(f"Rendering failed: {e}")
        raise RuntimeError(f"Rendering failed: {e}") from e


__all__ = ["render_layout"]
