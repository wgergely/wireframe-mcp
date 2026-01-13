"""PromptBuilder for RAG-enhanced LLM prompts.

Constructs prompts with schema definitions and few-shot examples
retrieved from the VectorStore.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.schema import export_llm_schema
from src.vector.serializer import LayoutSerializer

if TYPE_CHECKING:
    from src.mid import LayoutNode
    from src.vector import VectorStore


@dataclass
class PromptConfig:
    """Configuration for prompt building.

    Attributes:
        max_examples: Maximum number of few-shot examples to include.
        include_schema: Whether to include component schema.
        include_examples: Whether to include RAG examples.
        max_example_length: Max characters per serialized example.
    """

    max_examples: int = 3
    include_schema: bool = True
    include_examples: bool = True
    max_example_length: int = 500


@dataclass
class PromptContext:
    """Context for a generated prompt.

    Tracks what was included in the prompt for debugging/analysis.

    Attributes:
        query: Original user query.
        schema_included: Whether schema was included.
        example_count: Number of examples included.
        example_ids: IDs of included examples.
        total_tokens_estimate: Rough token count estimate.
    """

    query: str
    schema_included: bool = False
    example_count: int = 0
    example_ids: list[str] = field(default_factory=list)
    total_tokens_estimate: int = 0


class PromptBuilder:
    """Builds LLM prompts with RAG-enhanced context.

    Queries VectorStore for similar layouts and injects them
    as few-shot examples alongside schema definitions.

    Example:
        >>> from src.vector import VectorStore
        >>> store = VectorStore()
        >>> store.load(Path("data/index"))
        >>> builder = PromptBuilder(store)
        >>> prompt = builder.build("login form with email and password")
        >>> print(prompt)
    """

    def __init__(
        self,
        vector_store: "VectorStore | None" = None,
        config: PromptConfig | None = None,
    ):
        """Initialize PromptBuilder.

        Args:
            vector_store: VectorStore for retrieving similar layouts.
            config: Prompt building configuration.
        """
        self._store = vector_store
        self._config = config or PromptConfig()
        self._serializer = LayoutSerializer()

    def build(self, query: str) -> str:
        """Build a complete prompt for LLM generation.

        Args:
            query: User's natural language layout request.

        Returns:
            Formatted prompt with schema, examples, and query.
        """
        prompt, _ = self.build_with_context(query)
        return prompt

    def build_with_context(self, query: str) -> tuple[str, PromptContext]:
        """Build prompt and return context metadata.

        Args:
            query: User's natural language layout request.

        Returns:
            Tuple of (prompt_string, PromptContext).
        """
        parts: list[str] = []
        context = PromptContext(query=query)

        # 1. Schema injection
        if self._config.include_schema:
            schema_section = self._format_schema()
            parts.append(schema_section)
            context.schema_included = True

        # 2. RAG examples
        if self._config.include_examples and self._store is not None:
            examples_section, example_ids = self._format_examples(query)
            if examples_section:
                parts.append(examples_section)
                context.example_count = len(example_ids)
                context.example_ids = example_ids

        # 3. User query
        parts.append(self._format_query(query))

        prompt = "\n\n".join(parts)
        context.total_tokens_estimate = len(prompt) // 4  # Rough estimate
        return prompt, context

    def _format_schema(self) -> str:
        """Format the component schema section."""
        schema = export_llm_schema()
        return f"""## UI Component Schema

You must use only these component types in your layouts:

{schema}
"""

    def _format_examples(self, query: str) -> tuple[str, list[str]]:
        """Format the few-shot examples section.

        Args:
            query: User query for similarity search.

        Returns:
            Tuple of (formatted_section, list_of_example_ids).
        """
        if self._store is None:
            return "", []

        results = self._store.search(query, k=self._config.max_examples)
        if not results:
            return "", []

        examples: list[str] = []
        example_ids: list[str] = []

        for r in results:
            metadata = self._store.get_metadata(r.id)
            if metadata and "text" in metadata:
                text = metadata["text"]
                if len(text) > self._config.max_example_length:
                    text = text[: self._config.max_example_length] + "..."
                examples.append(f"### Example: {r.id}\n```\n{text}\n```")
                example_ids.append(r.id)

        if not examples:
            return "", []

        return f"""## Similar Layout Examples

These are real UI layouts similar to your request:

{chr(10).join(examples)}
""", example_ids

    def _format_query(self, query: str) -> str:
        """Format the user query section."""
        return f"""## Your Task

Generate a UI layout in JSON format for this request:

"{query}"

Respond with a valid JSON LayoutNode structure.
"""


__all__ = [
    "PromptBuilder",
    "PromptConfig",
    "PromptContext",
]
