"""Refine layout tool for iterative improvements."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def refine_layout(
    artifact_id: str,
    feedback: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Refine an existing layout based on user feedback.

    Takes a previous layout and natural language feedback to generate
    an improved version. Creates a parent-child link for lineage tracking.

    Args:
        artifact_id: ID of the layout to refine.
        feedback: Natural language description of desired changes.
            Examples:
            - "move the sidebar to the right"
            - "add a search box to the header"
            - "make the buttons larger"
        model: LLM model to use (optional).
        temperature: Creativity level 0.0-2.0. Default: 0.7

    Returns:
        Dictionary with:
        - artifact_id: New artifact ID
        - layout: Refined layout JSON
        - draft: Updated text tree
        - stats: Generation statistics
        - parent_id: ID of the original layout

    Example:
        >>> original = generate_layout("login form")
        >>> refined = refine_layout(
        ...     original["artifact_id"],
        ...     "add forgot password link below the form"
        ... )
    """
    from src.history import get_history_manager

    from .generate import generate_layout

    # Get original artifact
    manager = get_history_manager()
    original = manager.get_artifact(artifact_id)

    if not original:
        raise ValueError(f"Artifact not found: {artifact_id}")

    # Build refinement query that includes original context
    refinement_query = (
        f"Refine this layout: {original.query}\n\n"
        f"Current structure:\n{original.draft}\n\n"
        f"User feedback: {feedback}\n\n"
        f"Generate an updated layout incorporating the feedback."
    )

    # Generate with parent link
    result = generate_layout(
        query=refinement_query,
        model=model,
        temperature=temperature,
        include_rag=True,
        persist=True,
        session_id=original.session_id,
        parent_id=artifact_id,
    )

    # Log refinement interaction with feedback stored separately
    manager.store_interaction(
        tool_name="refine_layout",
        request_params={
            "artifact_id": artifact_id,
            "model": model,
            "temperature": temperature,
        },
        session_id=original.session_id,
        artifact_id=result.get("artifact_id"),
        feedback=feedback,  # Store feedback separately for later retrieval
        response_summary={
            "status": "success",
            "parent_id": artifact_id,
        },
    )

    result["parent_id"] = artifact_id
    return result


__all__ = ["refine_layout"]
