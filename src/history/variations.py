"""Variation generation engine for wireframe-mcp.

Provides functionality to generate multiple layout variations
with diversity metrics and ranking.
"""

import logging
from typing import Any
from uuid import uuid4

import numpy as np

from .lib import HistoryManager
from .models import (
    GenerationArtifact,
    VariationRequest,
    VariationSet,
)

logger = logging.getLogger(__name__)


def calculate_temperature_spread(
    count: int,
    temperature_range: tuple[float, float] | None = None,
) -> list[float]:
    """Calculate temperature values for variation generation.

    Args:
        count: Number of variations.
        temperature_range: Optional (min, max) range.

    Returns:
        List of temperature values.
    """
    if temperature_range:
        low, high = temperature_range
    else:
        # Default spread around 0.7
        low, high = 0.5, 1.0

    if count == 1:
        return [(low + high) / 2]

    return list(np.linspace(low, high, count))


def calculate_structural_diversity(artifacts: list[GenerationArtifact]) -> float:
    """Calculate diversity score based on structural differences.

    Measures how different the layouts are from each other.
    Score ranges from 0 (identical) to 1 (maximally different).

    Args:
        artifacts: List of artifacts to compare.

    Returns:
        Diversity score (0-1).
    """
    if len(artifacts) < 2:
        return 0.0

    # Extract features for comparison
    features = []
    for artifact in artifacts:
        layout = artifact.layout
        feature = _extract_layout_features(layout)
        features.append(feature)

    # Calculate pairwise differences
    differences = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            diff = _feature_distance(features[i], features[j])
            differences.append(diff)

    # Average difference normalized to 0-1
    if differences:
        return min(1.0, sum(differences) / len(differences))
    return 0.0


def _extract_layout_features(layout: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable features from a layout.

    Args:
        layout: Layout JSON dict.

    Returns:
        Feature dictionary.
    """
    features: dict[str, Any] = {
        "node_count": 0,
        "max_depth": 0,
        "component_types": set(),
        "has_children": False,
        "orientation": layout.get("orientation", "vertical"),
    }

    def traverse(node: dict[str, Any], depth: int = 0) -> None:
        features["node_count"] += 1
        features["max_depth"] = max(features["max_depth"], depth)

        node_type = node.get("type", "unknown")
        features["component_types"].add(node_type)

        children = node.get("children", [])
        if children:
            features["has_children"] = True
            for child in children:
                traverse(child, depth + 1)

    traverse(layout)

    # Convert set to sorted list for comparison
    features["component_types"] = sorted(features["component_types"])

    return features


def _feature_distance(f1: dict[str, Any], f2: dict[str, Any]) -> float:
    """Calculate distance between two feature sets.

    Args:
        f1: First feature set.
        f2: Second feature set.

    Returns:
        Distance score (0-1).
    """
    scores = []

    # Node count difference (normalized)
    max_nodes = max(f1["node_count"], f2["node_count"], 1)
    node_diff = abs(f1["node_count"] - f2["node_count"]) / max_nodes
    scores.append(node_diff)

    # Depth difference (normalized)
    max_depth = max(f1["max_depth"], f2["max_depth"], 1)
    depth_diff = abs(f1["max_depth"] - f2["max_depth"]) / max_depth
    scores.append(depth_diff)

    # Component type overlap (Jaccard distance)
    types1 = set(f1["component_types"])
    types2 = set(f2["component_types"])
    if types1 or types2:
        intersection = len(types1 & types2)
        union = len(types1 | types2)
        type_diff = 1 - (intersection / union if union else 0)
        scores.append(type_diff)

    # Orientation difference
    orientation_diff = 0.0 if f1["orientation"] == f2["orientation"] else 0.5
    scores.append(orientation_diff)

    return sum(scores) / len(scores) if scores else 0.0


def rank_by_complexity(artifacts: list[GenerationArtifact]) -> list[str]:
    """Rank artifacts by structural complexity (most complex first).

    Args:
        artifacts: Artifacts to rank.

    Returns:
        Ordered list of artifact IDs.
    """
    scored = []
    for artifact in artifacts:
        features = _extract_layout_features(artifact.layout)
        # Complexity = nodes + depth + component variety
        complexity = (
            features["node_count"]
            + features["max_depth"] * 2
            + len(features["component_types"])
        )
        scored.append((artifact.id, complexity))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored]


def rank_by_depth(artifacts: list[GenerationArtifact]) -> list[str]:
    """Rank artifacts by layout depth (deepest first).

    Args:
        artifacts: Artifacts to rank.

    Returns:
        Ordered list of artifact IDs.
    """
    scored = []
    for artifact in artifacts:
        features = _extract_layout_features(artifact.layout)
        scored.append((artifact.id, features["max_depth"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored]


def rank_by_component_variety(artifacts: list[GenerationArtifact]) -> list[str]:
    """Rank artifacts by component type variety (most variety first).

    Args:
        artifacts: Artifacts to rank.

    Returns:
        Ordered list of artifact IDs.
    """
    scored = []
    for artifact in artifacts:
        features = _extract_layout_features(artifact.layout)
        scored.append((artifact.id, len(features["component_types"])))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored]


def rank_by_node_count(artifacts: list[GenerationArtifact]) -> list[str]:
    """Rank artifacts by node count (most nodes first).

    Args:
        artifacts: Artifacts to rank.

    Returns:
        Ordered list of artifact IDs.
    """
    scored = []
    for artifact in artifacts:
        features = _extract_layout_features(artifact.layout)
        scored.append((artifact.id, features["node_count"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored]


def generate_rankings(
    artifacts: list[GenerationArtifact],
) -> dict[str, list[str]]:
    """Generate all ranking criteria for artifacts.

    Args:
        artifacts: Artifacts to rank.

    Returns:
        Dict mapping criterion name to ordered artifact IDs.
    """
    return {
        "complexity": rank_by_complexity(artifacts),
        "depth": rank_by_depth(artifacts),
        "variety": rank_by_component_variety(artifacts),
        "size": rank_by_node_count(artifacts),
    }


async def generate_variations_async(
    request: VariationRequest,
    history: HistoryManager,
    generator_fn: Any,  # Callable that generates a layout
    session_id: str | None = None,
) -> VariationSet:
    """Generate N layout variations asynchronously.

    Args:
        request: Variation request with query and count.
        history: History manager for storage.
        generator_fn: Async function that generates a single layout.
            Should accept (query, temperature, model, include_rag) and return
            (layout_dict, draft_str, stats, rag_example_ids, rag_scores).
        session_id: Session to attach variations to.

    Returns:
        VariationSet with generated artifacts.
    """
    import asyncio

    # Get session
    if session_id is None:
        session = history.get_or_create_session()
        session_id = session.id

    # Calculate temperature spread
    temperatures = calculate_temperature_spread(
        request.count,
        request.temperature_range,
    )

    # Generate variation group ID
    variation_group = str(uuid4())

    # Generate all variations concurrently
    async def generate_one(temp: float, index: int) -> GenerationArtifact:
        layout, draft, stats, rag_ids, rag_scores = await generator_fn(
            query=request.query,
            temperature=temp,
            model=request.model,
            include_rag=request.include_rag,
        )

        return history.store_artifact(
            query=request.query,
            layout=layout,
            draft=draft,
            session_id=session_id,
            model=request.model or "",
            temperature=temp,
            stats=stats,
            rag_example_ids=rag_ids,
            rag_scores=rag_scores,
            variation_group=variation_group,
            variation_index=index,
        )

    artifacts = await asyncio.gather(
        *[generate_one(temp, i) for i, temp in enumerate(temperatures)]
    )

    # Calculate metrics
    diversity = calculate_structural_diversity(list(artifacts))
    rankings = generate_rankings(list(artifacts))

    # Store variation set
    variation_set = history.store_variation_set(
        request=request,
        artifact_ids=[a.id for a in artifacts],
        diversity_score=diversity,
        rankings=rankings,
        session_id=session_id,
    )

    logger.info(f"Generated {len(artifacts)} variations with diversity={diversity:.2f}")

    return variation_set


def generate_variations_sync(
    request: VariationRequest,
    history: HistoryManager,
    generator_fn: Any,  # Callable that generates a layout
    session_id: str | None = None,
) -> VariationSet:
    """Generate N layout variations synchronously.

    Args:
        request: Variation request with query and count.
        history: History manager for storage.
        generator_fn: Function that generates a single layout.
            Should accept (query, temperature, model, include_rag) and return
            (layout_dict, draft_str, stats, rag_example_ids, rag_scores).
        session_id: Session to attach variations to.

    Returns:
        VariationSet with generated artifacts.
    """
    # Get session
    if session_id is None:
        session = history.get_or_create_session()
        session_id = session.id

    # Calculate temperature spread
    temperatures = calculate_temperature_spread(
        request.count,
        request.temperature_range,
    )

    # Generate variation group ID
    variation_group = str(uuid4())

    # Generate all variations sequentially
    artifacts = []
    for i, temp in enumerate(temperatures):
        layout, draft, stats, rag_ids, rag_scores = generator_fn(
            query=request.query,
            temperature=temp,
            model=request.model,
            include_rag=request.include_rag,
        )

        artifact = history.store_artifact(
            query=request.query,
            layout=layout,
            draft=draft,
            session_id=session_id,
            model=request.model or "",
            temperature=temp,
            stats=stats,
            rag_example_ids=rag_ids,
            rag_scores=rag_scores,
            variation_group=variation_group,
            variation_index=i,
        )
        artifacts.append(artifact)

    # Calculate metrics
    diversity = calculate_structural_diversity(artifacts)
    rankings = generate_rankings(artifacts)

    # Store variation set
    variation_set = history.store_variation_set(
        request=request,
        artifact_ids=[a.id for a in artifacts],
        diversity_score=diversity,
        rankings=rankings,
        session_id=session_id,
    )

    logger.info(f"Generated {len(artifacts)} variations with diversity={diversity:.2f}")

    return variation_set


def compare_artifacts(
    artifacts: list[GenerationArtifact],
) -> dict[str, Any]:
    """Compare multiple artifacts.

    Args:
        artifacts: Artifacts to compare.

    Returns:
        Comparison results including diversity and rankings.
    """
    if not artifacts:
        return {
            "count": 0,
            "diversity_score": 0.0,
            "rankings": {},
            "feature_summary": {},
        }

    diversity = calculate_structural_diversity(artifacts)
    rankings = generate_rankings(artifacts)

    # Feature summary
    all_features = [_extract_layout_features(a.layout) for a in artifacts]
    feature_summary = {
        "node_counts": [f["node_count"] for f in all_features],
        "max_depths": [f["max_depth"] for f in all_features],
        "all_component_types": sorted(
            set().union(*[set(f["component_types"]) for f in all_features])
        ),
        "avg_node_count": sum(f["node_count"] for f in all_features)
        / len(all_features),
        "avg_depth": sum(f["max_depth"] for f in all_features) / len(all_features),
    }

    return {
        "count": len(artifacts),
        "diversity_score": diversity,
        "rankings": rankings,
        "feature_summary": feature_summary,
    }


__all__ = [
    "calculate_temperature_spread",
    "calculate_structural_diversity",
    "rank_by_complexity",
    "rank_by_depth",
    "rank_by_component_variety",
    "rank_by_node_count",
    "generate_rankings",
    "generate_variations_async",
    "generate_variations_sync",
    "compare_artifacts",
]
