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

    Returns:
        Dictionary containing:
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

        return {
            "layout": layout_dict,
            "draft": draft,
            "stats": {
                "attempts": output.stats.attempts,
                "tokens": output.stats.total_tokens,
                "model": output.stats.final_model,
                "rag_examples": len(output.prompt_context.examples),
            },
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


__all__ = ["generate_layout"]
