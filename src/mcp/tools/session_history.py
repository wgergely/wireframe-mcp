"""Session history tools for MCP server.

Provides tools to reconstruct conversation context and audit agent behavior.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_session_history(
    session_id: str | None = None,
    limit: int = 50,
    include_artifacts: bool = True,
) -> dict[str, Any]:
    """Get session history for conversation reconstruction.

    Returns a chronological timeline of all tool calls and generated
    artifacts for a session. Use this to understand what has happened
    in a design conversation.

    Args:
        session_id: Session to get history for (uses current if None).
        limit: Maximum events to return. Default: 50
        include_artifacts: Include artifact details in timeline. Default: True

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - session_name: Session name if set
        - timeline: List of events in chronological order
        - artifact_count: Total artifacts in session
        - interaction_count: Total interactions logged
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    # Get or use current session
    if session_id is None:
        session = manager.get_or_create_session()
        session_id = session.id
    else:
        session = manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

    # Get timeline
    timeline = manager.get_session_timeline(
        session_id=session_id,
        include_artifacts=include_artifacts,
    )

    # Apply limit
    if limit and len(timeline) > limit:
        timeline = timeline[-limit:]  # Most recent

    # Get counts
    interactions = manager.list_interactions(session_id=session_id, limit=10000)
    artifacts = manager.list_artifacts(session_id=session_id, limit=10000)

    return {
        "session_id": session_id,
        "session_name": session.name,
        "timeline": timeline,
        "artifact_count": len(artifacts),
        "interaction_count": len(interactions),
    }


def get_artifact_history(
    artifact_id: str,
) -> dict[str, Any]:
    """Get the full history of an artifact including all refinements.

    Traces the lineage of an artifact and returns all interactions
    that led to its creation, including feedback given at each step.

    Args:
        artifact_id: Artifact to get history for.

    Returns:
        Dictionary containing:
        - artifact_id: The artifact ID
        - lineage: List of artifacts from root to this one
        - interactions: All interactions related to this lineage
        - feedback_chain: List of feedback strings in order
    """
    from src.history import get_history_manager

    manager = get_history_manager()

    artifact = manager.get_artifact(artifact_id)
    if not artifact:
        raise ValueError(f"Artifact not found: {artifact_id}")

    # Get lineage
    lineage_data = manager.get_lineage(artifact_id)
    ancestors = lineage_data["ancestors"]

    # Build lineage chain (root to current)
    lineage_chain = list(reversed(ancestors)) + [artifact]

    # Get all interactions for artifacts in lineage
    all_interactions = []
    feedback_chain = []

    for art in lineage_chain:
        interactions = manager.list_interactions(artifact_id=art.id)
        for interaction in interactions:
            all_interactions.append(
                {
                    "artifact_id": art.id,
                    "tool_name": interaction.tool_name,
                    "timestamp": interaction.created_at.isoformat(),
                    "feedback": interaction.feedback,
                    "request_params": interaction.request_params,
                }
            )
            if interaction.feedback:
                feedback_chain.append(interaction.feedback)

    return {
        "artifact_id": artifact_id,
        "lineage": [
            {
                "id": art.id,
                "query": art.query,
                "draft_preview": art.draft[:100] + "..."
                if len(art.draft) > 100
                else art.draft,
                "parent_id": art.parent_id,
                "created_at": art.created_at.isoformat(),
            }
            for art in lineage_chain
        ],
        "interactions": all_interactions,
        "feedback_chain": feedback_chain,
    }


__all__ = [
    "get_session_history",
    "get_artifact_history",
]
