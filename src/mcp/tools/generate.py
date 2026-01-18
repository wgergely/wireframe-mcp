"""Generate layout tool for MCP server.

This tool generates UI layouts from natural language descriptions,
returning structured JSON and a draft text tree for quick review.
"""

import logging
from typing import Any

from src.output import format_layout_tree

logger = logging.getLogger(__name__)


def generate_layout(
    query: str,
    model: str | None = None,
    temperature: float = 0.7,
    provider: str = "d2",
    include_rag: bool = True,
    persist: bool = True,
    session_id: str | None = None,
    parent_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a UI layout from natural language description.

    This is the primary creation tool. It returns structured JSON and a
    draft text tree for quick human review. Use preview_layout to get
    a visual wireframe image.

    Args:
        query: Natural language description of the desired layout.
            Examples:
            - "login form with email and password"
            - "dashboard with sidebar and main content area"
            - "settings page with toggle switches"
        model: LLM model to use. Options include:
            - "gpt-4.1-mini" (OpenAI, default)
            - "claude-sonnet-4-5" (Anthropic)
            - "deepseek-v3" (DeepSeek)
            If not specified, uses the first available provider.
        temperature: Generation temperature (0.0-2.0). Higher values
            produce more creative/varied outputs. Default: 0.7
        provider: Target DSL provider for rendering hints.
            Options: "d2", "plantuml". Default: "d2"
        include_rag: Whether to include similar layouts from vector
            store as context for generation. Default: True
        persist: Whether to save to history. Default: True
        session_id: Session to attach artifact to (optional).
        parent_id: Parent artifact ID for refinements (optional).
        tags: User-defined tags for organization (optional).

    Returns:
        Dictionary containing:
        - artifact_id: UUID of stored artifact (if persist=True)
        - layout: The generated LayoutNode as a JSON dict
        - draft: Human-readable text tree for quick review
        - stats: Generation statistics (attempts, tokens, model used)

    Raises:
        ToolError: If generation fails after all retries.

    Example:
        >>> result = generate_layout("login form with email and password")
        >>> print(result["draft"])
        Login [container]
        └── Form [card]
            ├── Email [input]
            ├── Password [input]
            └── Submit [button]
        >>> print(f"Artifact ID: {result['artifact_id']}")
    """
    from src.config import get_index_dir
    from src.llm import GeneratorConfig, LayoutGenerator, LLMModel, create_llm_backend

    try:
        # Create backend if model specified
        backend = None
        if model:
            llm_model = LLMModel.by_name(model)
            if llm_model is None:
                available = [m.spec.name for m in LLMModel]
                raise ValueError(
                    f"Unknown model: {model}. Available: {', '.join(available)}"
                )
            backend = create_llm_backend(llm_model)

        # Load vector store if RAG enabled
        vector_store = None
        if include_rag:
            try:
                from src.vector import VectorStore

                index_dir = get_index_dir()
                if (index_dir / "index.faiss").exists():
                    vector_store = VectorStore()
                    vector_store.load(index_dir)
                    logger.info(f"Loaded RAG index with {len(vector_store)} items")
                else:
                    logger.debug("No RAG index found, generating without context")
            except Exception as e:
                logger.warning(f"Could not load RAG index: {e}")

        # Configure generator
        config = GeneratorConfig(
            temperature=temperature,
            max_retries=3,
            include_rag_context=vector_store is not None,
        )

        # Generate layout
        generator = LayoutGenerator(
            backend=backend,
            vector_store=vector_store,
            config=config,
        )

        logger.info(f"Generating layout for: {query}")
        output = generator.generate(query, target_provider=provider)

        # Build response
        layout_dict = output.context.node.model_dump(mode="json")
        draft = format_layout_tree(output.context.node)

        result: dict[str, Any] = {
            "layout": layout_dict,
            "draft": draft,
            "stats": {
                "attempts": output.stats.attempts,
                "tokens": output.stats.total_tokens,
                "model": output.stats.final_model,
                "rag_examples": len(output.prompt_context.examples),
            },
        }

        # Persist to history if enabled
        if persist:
            try:
                from src.history import GenerationStats, get_history_manager

                manager = get_history_manager()
                artifact = manager.store_artifact(
                    query=query,
                    layout=layout_dict,
                    draft=draft,
                    session_id=session_id,
                    parent_id=parent_id,
                    model=output.stats.final_model,
                    temperature=temperature,
                    provider=provider,
                    stats=GenerationStats(
                        attempts=output.stats.attempts,
                        validation_retries=output.stats.validation_retries,
                        json_repairs=output.stats.json_repairs,
                        total_tokens=output.stats.total_tokens,
                        final_model=output.stats.final_model,
                    ),
                    rag_example_ids=output.prompt_context.example_ids,
                    tags=tags,
                )
                result["artifact_id"] = artifact.id
                logger.debug(f"Persisted artifact: {artifact.id}")
            except Exception as e:
                logger.warning(f"Failed to persist artifact: {e}")
                result["artifact_id"] = None

        return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


def generate_variations(
    query: str,
    count: int = 3,
    temperature_range: tuple[float, float] | None = None,
    model: str | None = None,
    include_rag: bool = True,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Generate multiple layout variations for comparison.

    Creates N different layouts from the same query, varying temperature
    to produce diversity. Useful for exploring design alternatives.

    Args:
        query: Natural language description of the desired layout.
        count: Number of variations to generate (1-10). Default: 3
        temperature_range: Optional (min, max) temperature spread.
            Default: (0.5, 1.0)
        model: LLM model to use (optional).
        include_rag: Whether to include RAG context. Default: True
        session_id: Session to attach variations to (optional).

    Returns:
        Dictionary containing:
        - variation_set_id: UUID of the variation set
        - artifacts: List of artifact summaries with IDs and drafts
        - diversity_score: How different the variations are (0-1)
        - rankings: Artifacts ranked by different criteria

    Example:
        >>> result = generate_variations("settings page", count=3)
        >>> print(f"Diversity: {result['diversity_score']:.2f}")
        >>> for artifact in result["artifacts"]:
        ...     print(f"  {artifact['id']}: temp={artifact['temperature']}")
    """
    from src.history import (
        VariationRequest,
        generate_variations_sync,
        get_history_manager,
    )

    # Validate count
    if not 1 <= count <= 10:
        raise ValueError(f"count must be 1-10, got {count}")

    manager = get_history_manager()

    # Create generator function that matches expected signature
    def generator_fn(
        query: str,
        temperature: float,
        model: str | None,
        include_rag: bool,
    ) -> tuple:
        """Generate a single layout for variation set."""
        result = generate_layout(
            query=query,
            model=model,
            temperature=temperature,
            include_rag=include_rag,
            persist=False,  # We'll persist via variation set
        )
        # Return tuple expected by variation engine
        from src.history import GenerationStats

        stats = GenerationStats(
            attempts=result["stats"]["attempts"],
            total_tokens=result["stats"]["tokens"],
            final_model=result["stats"]["model"],
        )
        return (
            result["layout"],
            result["draft"],
            stats,
            [],  # rag_example_ids
            {},  # rag_scores
        )

    # Create variation request
    request = VariationRequest(
        query=query,
        count=count,
        temperature_range=temperature_range,
        model=model,
        include_rag=include_rag,
    )

    # Generate variations
    variation_set = generate_variations_sync(
        request=request,
        history=manager,
        generator_fn=generator_fn,
        session_id=session_id,
    )

    # Get artifacts for response
    artifacts = manager.get_variation_artifacts(variation_set.id)

    return {
        "variation_set_id": variation_set.id,
        "artifacts": [
            {
                "id": a.id,
                "draft_preview": a.draft[:200] + "..."
                if len(a.draft) > 200
                else a.draft,
                "temperature": a.temperature,
                "variation_index": a.variation_index,
            }
            for a in artifacts
        ],
        "diversity_score": variation_set.diversity_score,
        "rankings": variation_set.rankings,
    }


__all__ = ["generate_layout", "generate_variations"]
