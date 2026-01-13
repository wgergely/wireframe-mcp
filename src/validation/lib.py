"""Layout validation and static analysis.

This module provides validation functions for LayoutNode trees,
detecting structural issues before transpilation.
"""

from dataclasses import dataclass

from src.ir import LayoutNode


@dataclass
class ValidationError:
    """Represents a validation error in a layout tree.

    Attributes:
        node_id: ID of the node with the error.
        message: Human-readable error description.
        error_type: Category of the error.
    """

    node_id: str
    message: str
    error_type: str


def validate_layout(node: LayoutNode) -> list[ValidationError]:
    """Validate a LayoutNode tree for structural issues.

    Performs the following checks:
        - Unique ID enforcement (no duplicate IDs)
        - Flex ratio range validation (1-12)
        - Cycle detection (no node is ancestor of itself)

    Args:
        node: The root LayoutNode to validate.

    Returns:
        list[ValidationError]: List of validation errors (empty if valid).

    Example:
        >>> errors = validate_layout(root_node)
        >>> if errors:
        ...     for e in errors:
        ...         print(f"{e.node_id}: {e.message}")
    """
    errors: list[ValidationError] = []

    # Collect all IDs and check for duplicates
    id_counts: dict[str, int] = {}
    _collect_ids(node, id_counts)

    for node_id, count in id_counts.items():
        if count > 1:
            errors.append(
                ValidationError(
                    node_id=node_id,
                    message=f"Duplicate ID '{node_id}' appears {count} times",
                    error_type="duplicate_id",
                )
            )

    # Validate flex ratios
    errors.extend(_validate_flex_ratios(node))

    # Check for cycles (shouldn't happen with proper tree structure, but check anyway)
    errors.extend(_detect_cycles(node))

    return errors


def is_valid(node: LayoutNode) -> bool:
    """Check if a layout tree is valid.

    Convenience function that returns True if no validation errors exist.

    Args:
        node: The root LayoutNode to validate.

    Returns:
        bool: True if the tree is valid, False otherwise.

    Example:
        >>> if is_valid(root_node):
        ...     output = provider.transpile(root_node)
    """
    return not validate_layout(node)


def _collect_ids(node: LayoutNode, id_counts: dict[str, int]) -> None:
    """Recursively collect all node IDs and count occurrences.

    Args:
        node: Current node to process.
        id_counts: Accumulator dictionary mapping ID to count.
    """
    id_counts[node.id] = id_counts.get(node.id, 0) + 1
    for child in node.children:
        _collect_ids(child, id_counts)


def _validate_flex_ratios(node: LayoutNode) -> list[ValidationError]:
    """Validate flex ratios are within bounds.

    Note: Pydantic already enforces this, but we check anyway for
    cases where the model is constructed from raw dicts.

    Args:
        node: The root node to validate.

    Returns:
        list[ValidationError]: Flex ratio errors found.
    """
    errors: list[ValidationError] = []

    def _check(n: LayoutNode) -> None:
        if n.flex_ratio < 1 or n.flex_ratio > 12:
            errors.append(
                ValidationError(
                    node_id=n.id,
                    message=f"flex_ratio {n.flex_ratio} outside valid range 1-12",
                    error_type="invalid_flex_ratio",
                )
            )
        for child in n.children:
            _check(child)

    _check(node)
    return errors


def _detect_cycles(node: LayoutNode) -> list[ValidationError]:
    """Detect cycles in the tree structure.

    With Pydantic models this shouldn't occur, but we check for
    safety when trees are constructed programmatically.

    Args:
        node: The root node to check.

    Returns:
        list[ValidationError]: Cycle errors found.
    """
    errors: list[ValidationError] = []
    visited: set[int] = set()

    def _check(n: LayoutNode, path: set[int]) -> None:
        obj_id = id(n)
        if obj_id in path:
            errors.append(
                ValidationError(
                    node_id=n.id,
                    message=f"Cycle detected: node '{n.id}' is its own ancestor",
                    error_type="cycle",
                )
            )
            return
        if obj_id in visited:
            return
        visited.add(obj_id)
        path.add(obj_id)
        for child in n.children:
            _check(child, path)
        path.remove(obj_id)

    _check(node, set())
    return errors
