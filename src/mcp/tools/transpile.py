"""Transpile layout tool for MCP server.

This tool converts layout JSON to DSL code (D2 or PlantUML).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def transpile_layout(
    layout: dict[str, Any],
    provider: str = "d2",
) -> dict[str, Any]:
    """Convert a layout JSON to DSL code.

    Transpiles the layout structure to a diagram DSL that can be
    rendered by external tools or used as documentation.

    Args:
        layout: Layout JSON to transpile. Should be the output from
            generate_layout or a manually constructed layout dict.
        provider: Target DSL provider. Options:
            - "d2": D2 diagram language (default)
            - "plantuml": PlantUML Salt wireframe notation

    Returns:
        Dictionary containing:
        - dsl_code: The transpiled DSL code string
        - provider: Provider name used
        - line_count: Number of lines in output

    Example:
        >>> result = transpile_layout(layout, provider="d2")
        >>> print(result["dsl_code"])
        root: Login Form {
          form: {
            email: Input
            password: Input
            submit: Button
          }
        }
    """
    from pydantic import ValidationError as PydanticValidationError

    from src.mid import LayoutNode
    from src.providers import get_provider, list_providers

    # Validate provider
    available = list_providers()
    if provider not in available:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {', '.join(available)}"
        )

    # Parse layout
    try:
        node = LayoutNode.model_validate(layout)
    except PydanticValidationError as e:
        raise ValueError(f"Invalid layout structure: {e}") from e

    # Transpile
    dsl_provider = get_provider(provider)
    dsl_code = dsl_provider.transpile(node)

    return {
        "dsl_code": dsl_code,
        "provider": provider,
        "line_count": len(dsl_code.strip().split("\n")),
    }


__all__ = ["transpile_layout"]
