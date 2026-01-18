"""Validate layout tool for MCP server.

This tool validates layout structures for errors and provides
structural statistics.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _count_nodes(node: Any, depth: int = 0) -> tuple[int, int, set[str]]:
    """Count nodes, max depth, and component types in a layout."""
    count = 1
    max_depth = depth
    types = {node.get("type", "unknown") if isinstance(node, dict) else str(node.type)}

    children = node.get("children", []) if isinstance(node, dict) else node.children
    for child in children:
        child_count, child_depth, child_types = _count_nodes(child, depth + 1)
        count += child_count
        max_depth = max(max_depth, child_depth)
        types.update(child_types)

    return count, max_depth, types


def validate_layout(
    layout: dict[str, Any],
) -> dict[str, Any]:
    """Validate a layout structure for errors.

    Checks the layout for structural issues like duplicate IDs,
    invalid flex ratios, cycles, and component-specific constraints.

    Args:
        layout: Layout JSON to validate. Should be the output from
            generate_layout or a manually constructed layout dict.

    Returns:
        Dictionary containing:
        - valid: Boolean indicating if layout passes all checks
        - errors: List of error objects with node_id, message, error_type
        - warnings: List of warning objects (non-blocking issues)
        - stats: Structural statistics (node_count, max_depth, component_types)

    Example:
        >>> result = validate_layout({"id": "root", "type": "container", ...})
        >>> if not result["valid"]:
        ...     for error in result["errors"]:
        ...         print(f"Error in {error['node_id']}: {error['message']}")
    """
    from pydantic import ValidationError as PydanticValidationError

    from src.mid import LayoutNode
    from src.mid import validate_layout as mid_validate

    errors = []
    warnings = []

    # First, try to parse as LayoutNode
    try:
        node = LayoutNode.model_validate(layout)
    except PydanticValidationError as e:
        # Schema validation failed
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(
                {
                    "node_id": loc or "root",
                    "message": err["msg"],
                    "error_type": "schema_validation",
                }
            )
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "node_count": 0,
                "max_depth": 0,
                "component_types": [],
            },
        }

    # Run structural validation
    validation_errors = mid_validate(node)

    for err in validation_errors:
        errors.append(
            {
                "node_id": err.node_id,
                "message": err.message,
                "error_type": err.error_type,
            }
        )

    # Check for warnings (non-blocking issues)
    _collect_warnings(node, warnings)

    # Calculate stats from the dict (before parsing, to be safe)
    node_count, max_depth, component_types = _count_nodes(layout)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "node_count": node_count,
            "max_depth": max_depth,
            "component_types": sorted(component_types),
        },
    }


def _collect_warnings(node: Any, warnings: list[dict]) -> None:
    """Collect non-blocking warnings for a layout tree."""
    # Check for buttons without labels
    node_type = node.type if hasattr(node, "type") else node.get("type")
    node_id = node.id if hasattr(node, "id") else node.get("id")
    label = node.label if hasattr(node, "label") else node.get("label")

    if node_type in ("button", "link") and not label:
        warnings.append(
            {
                "node_id": node_id,
                "message": f"{node_type} has no label text",
                "warning_type": "missing_label",
            }
        )

    # Check for inputs without labels (accessibility)
    if node_type == "input" and not label:
        warnings.append(
            {
                "node_id": node_id,
                "message": "Input has no label (accessibility concern)",
                "warning_type": "accessibility",
            }
        )

    # Check for very deep nesting
    children = node.children if hasattr(node, "children") else node.get("children", [])

    # Recurse
    for child in children:
        _collect_warnings(child, warnings)


__all__ = ["validate_layout"]
