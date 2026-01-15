"""LayoutGenerator orchestrator for LLM-powered wireframe generation.

Integrates PromptBuilder, LLM backends, and validation to produce
validated LayoutNode trees from natural language queries.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.ir import TranspilationContext
from src.mid import LayoutNode, ValidationError, validate_layout
from src.prompt import PromptBuilder, PromptConfig, PromptContext

from ..backend import LLMBackend, create_llm_backend
from ..backend.base import GenerationConfig, InvalidResponseError, LLMError
from .retry import RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for LayoutGenerator.

    Attributes:
        max_retries: Maximum generation attempts before failing.
        retry_on_validation_error: Re-prompt with validation error feedback.
        repair_json: Attempt JSON repair on parse errors.
        temperature: LLM temperature for generation (0.0-2.0).
        include_rag_context: Include similar layouts from VectorStore.
        max_rag_examples: Maximum RAG examples to include in prompt.
    """

    max_retries: int = 3
    retry_on_validation_error: bool = True
    repair_json: bool = True
    temperature: float = 0.7
    include_rag_context: bool = True
    max_rag_examples: int = 3


@dataclass
class GenerationStats:
    """Statistics from layout generation.

    Attributes:
        attempts: Number of generation attempts made.
        validation_retries: Number of retries due to validation errors.
        json_repairs: Number of JSON repair attempts.
        total_tokens: Total tokens used across all attempts.
        final_model: Model identifier used for successful generation.
    """

    attempts: int = 0
    validation_retries: int = 0
    json_repairs: int = 0
    total_tokens: int = 0
    final_model: str = ""


@dataclass
class GenerationOutput:
    """Complete output from layout generation.

    Attributes:
        context: TranspilationContext with validated LayoutNode.
        prompt_context: PromptContext with prompt metadata.
        stats: Generation statistics.
        raw_response: Raw LLM response content.
    """

    context: TranspilationContext
    prompt_context: PromptContext
    stats: GenerationStats
    raw_response: str


# Type alias for VectorStore to avoid import cycle
VectorStore = Any


class LayoutGenerator:
    """Orchestrates LLM-based layout generation.

    Integrates PromptBuilder, LLM backend, and validation to produce
    validated LayoutNode trees from natural language queries.

    Pipeline:
        1. Build prompt with schema and RAG examples
        2. Generate JSON from LLM
        3. Parse and validate LayoutNode
        4. Retry with feedback on errors
        5. Return TranspilationContext

    Example:
        >>> from src.vector import VectorStore
        >>> store = VectorStore()
        >>> store.load(Path("data/index"))
        >>> generator = LayoutGenerator(vector_store=store)
        >>> output = generator.generate("login form with email and password")
        >>> print(output.context.node)

        >>> # Without RAG (simpler)
        >>> generator = LayoutGenerator()
        >>> output = generator.generate("simple dashboard with sidebar")
    """

    SYSTEM_PROMPT = """You are a UI layout architect. Generate valid JSON layouts.

RULES:
1. Output ONLY valid JSON matching the LayoutNode schema
2. Use ONLY component types from the provided schema
3. Every node MUST have unique "id" and valid "type"
4. flex_ratio must be integer 1-12 (12-column grid system)
5. Use descriptive IDs (e.g., "header", "login-form", "submit-btn")
6. orientation must be "horizontal", "vertical", or "overlay"

REQUIRED FIELDS:
- id: unique string identifier
- type: component type from schema

OPTIONAL FIELDS:
- label: text content
- flex_ratio: 1-12 (default: 1)
- orientation: layout direction (default: "vertical")
- children: nested LayoutNode array

LAYOUT PROPERTIES (optional - use for precise control):
- align: cross-axis alignment ("start", "center", "end", "stretch")
- justify: main-axis distribution ("start", "center", "end", "between", "around")
- gap: spacing between children in pixels (integer)
- padding: internal padding in pixels (integer)
- wrap: overflow behavior ("none", "wrap")

TEXT STYLING (optional - for text nodes):
- text_size: size hierarchy ("title", "heading", "body", "caption")
- text_weight: font weight ("light", "normal", "bold")
- text_transform: case ("none", "uppercase", "lowercase", "capitalize")
- text_align: alignment ("left", "center", "right")

OUTPUT FORMAT: Single JSON object with root LayoutNode structure.
Do not include any text before or after the JSON."""

    def __init__(
        self,
        backend: LLMBackend | None = None,
        vector_store: VectorStore | None = None,
        config: GeneratorConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ):
        """Initialize LayoutGenerator.

        Args:
            backend: LLM backend for generation. Creates default if None.
            vector_store: VectorStore for RAG examples (optional).
            config: Generator configuration.
            prompt_config: Prompt building configuration.
        """
        self._backend = backend or create_llm_backend()
        self._store = vector_store
        self._config = config or GeneratorConfig()
        self._prompt_builder = PromptBuilder(
            vector_store=vector_store,
            config=prompt_config
            or PromptConfig(
                max_examples=self._config.max_rag_examples,
                include_examples=self._config.include_rag_context,
            ),
        )
        self._retry_strategy = RetryStrategy(
            RetryConfig(
                max_retries=self._config.max_retries,
                repair_json=self._config.repair_json,
            )
        )

    def generate(
        self,
        query: str,
        *,
        target_provider: str = "d2",
    ) -> GenerationOutput:
        """Generate layout from natural language query.

        Args:
            query: Natural language layout description.
            target_provider: Target transpiler name ("d2", "plantuml").

        Returns:
            GenerationOutput with TranspilationContext and metadata.

        Raises:
            LLMError: If all generation attempts fail.
            InvalidResponseError: If response cannot be parsed as valid layout.
        """
        stats = GenerationStats()

        # Build prompt with RAG context
        prompt, prompt_context = self._prompt_builder.build_with_context(query)

        # Generation loop with retries
        last_error: Exception | None = None
        validation_feedback: str | None = None

        for attempt in range(self._config.max_retries):
            stats.attempts += 1

            try:
                # Add validation feedback to prompt if available
                effective_prompt = prompt
                if validation_feedback:
                    effective_prompt = (
                        f"{prompt}\n\n"
                        f"PREVIOUS ATTEMPT HAD ERRORS:\n{validation_feedback}\n"
                        "Please fix these issues in your response."
                    )
                    stats.validation_retries += 1

                # Generate from LLM
                gen_config = GenerationConfig(
                    temperature=self._config.temperature,
                    json_mode=True,
                )

                result = self._backend.generate(
                    effective_prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    config=gen_config,
                )

                stats.total_tokens += result.usage.get("total_tokens", 0)
                stats.final_model = result.model

                # Parse JSON response
                layout_dict = self._parse_response(result.content, stats)

                # Validate and create LayoutNode
                node = LayoutNode.model_validate(layout_dict)
                validation_errors = validate_layout(node)

                if validation_errors:
                    if (
                        self._config.retry_on_validation_error
                        and attempt < self._config.max_retries - 1
                    ):
                        validation_feedback = self._format_validation_errors(
                            validation_errors
                        )
                        logger.warning(
                            f"Validation errors on attempt {attempt + 1}, retrying: "
                            f"{validation_feedback}"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Layout has validation errors: {validation_errors}"
                        )

                # Success - build TranspilationContext
                similar_layouts = self._get_similar_layouts(query)

                context = TranspilationContext(
                    node=node,
                    similar_layouts=similar_layouts,
                    target_provider=target_provider,
                )

                logger.info(
                    f"Generated layout successfully after {stats.attempts} attempt(s)"
                )

                return GenerationOutput(
                    context=context,
                    prompt_context=prompt_context,
                    stats=stats,
                    raw_response=result.content,
                )

            except (json.JSONDecodeError, InvalidResponseError) as e:
                last_error = e
                logger.warning(f"Parse error on attempt {attempt + 1}: {e}")

            except LLMError as e:
                last_error = e
                logger.error(f"LLM error on attempt {attempt + 1}: {e}")
                if not self._retry_strategy.should_retry(e, attempt):
                    raise

        # All retries exhausted
        raise InvalidResponseError(
            f"Failed to generate valid layout after {stats.attempts} attempts. "
            f"Last error: {last_error}"
        )

    def _parse_response(
        self,
        content: str,
        stats: GenerationStats,
    ) -> dict[str, Any]:
        """Parse LLM response to JSON dict.

        Args:
            content: Raw response content.
            stats: Stats to update.

        Returns:
            Parsed dictionary.

        Raises:
            InvalidResponseError: If parsing fails.
        """
        content = content.strip()

        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try JSON repair if enabled
        if self._config.repair_json:
            stats.json_repairs += 1
            repaired = self._retry_strategy.repair_json(content)
            if repaired:
                logger.debug("JSON repair successful")
                return repaired

        raise InvalidResponseError(f"Cannot parse response as JSON: {content[:500]}...")

    def _format_validation_errors(
        self,
        errors: list[ValidationError],
    ) -> str:
        """Format validation errors for feedback prompt.

        Args:
            errors: List of validation errors.

        Returns:
            Formatted error string.
        """
        lines = ["Validation errors found:"]
        for error in errors:
            lines.append(f"- Node '{error.node_id}': {error.message}")
        return "\n".join(lines)

    def _get_similar_layouts(self, query: str) -> list[LayoutNode]:
        """Retrieve similar layouts from VectorStore.

        Currently returns empty list as VectorStore doesn't store
        full LayoutNode objects. Future enhancement could deserialize
        from stored JSON.

        Args:
            query: Query for similarity search.

        Returns:
            List of similar LayoutNode objects.
        """
        # TODO: Implement layout reconstruction from VectorStore
        # Currently VectorStore only stores serialized text, not full layouts
        return []


__all__ = [
    "LayoutGenerator",
    "GeneratorConfig",
    "GenerationStats",
    "GenerationOutput",
]
