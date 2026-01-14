"""Layout generation orchestrator.

Provides the LayoutGenerator class that integrates PromptBuilder,
LLM backends, and validation to produce layouts from natural language.
"""

from .lib import (
    GenerationOutput,
    GenerationStats,
    GeneratorConfig,
    LayoutGenerator,
)
from .retry import RetryConfig, RetryStrategy

__all__ = [
    "LayoutGenerator",
    "GeneratorConfig",
    "GenerationStats",
    "GenerationOutput",
    "RetryConfig",
    "RetryStrategy",
]
