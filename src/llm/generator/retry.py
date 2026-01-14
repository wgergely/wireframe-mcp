"""Retry strategies and error recovery for LLM generation.

Provides JSON repair utilities and retry logic for handling
transient errors and malformed responses.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry strategy.

    Attributes:
        max_retries: Maximum number of retry attempts.
        repair_json: Attempt to repair malformed JSON.
        exponential_backoff: Use exponential backoff for rate limits.
        initial_delay: Initial delay for backoff (seconds).
        max_delay: Maximum delay between retries (seconds).
    """

    max_retries: int = 3
    repair_json: bool = True
    exponential_backoff: bool = True
    initial_delay: float = 1.0
    max_delay: float = 30.0


class RetryStrategy:
    """Handles retries and error recovery for LLM generation.

    Provides:
    - JSON repair for common malformations
    - Exponential backoff delay calculation
    - Retry decision logic based on error type

    Example:
        >>> strategy = RetryStrategy()
        >>> repaired = strategy.repair_json('```json\\n{"key": "value"}\\n```')
        >>> print(repaired)
        {'key': 'value'}
    """

    # Common JSON repair patterns (pattern, replacement)
    JSON_REPAIR_PATTERNS: list[tuple[str, str]] = [
        # Remove markdown code blocks
        (r"^```json\s*", ""),
        (r"^```\s*", ""),
        (r"\s*```$", ""),
        # Fix trailing commas before closing braces/brackets
        (r",\s*}", "}"),
        (r",\s*]", "]"),
        # Fix unquoted keys (simple cases only)
        (r"(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":'),
    ]

    def __init__(self, config: RetryConfig | None = None):
        """Initialize retry strategy.

        Args:
            config: Retry configuration options.
        """
        self._config = config or RetryConfig()

    def repair_json(self, content: str) -> dict[str, Any] | None:
        """Attempt to repair malformed JSON.

        Tries various repair strategies:
        1. Strip markdown code blocks
        2. Fix trailing commas
        3. Extract JSON from mixed content
        4. Handle unquoted keys

        Args:
            content: Raw content that failed JSON parsing.

        Returns:
            Parsed dict if repair successful, None otherwise.
        """
        if not self._config.repair_json:
            return None

        # Try basic cleanup
        cleaned = content.strip()

        # Apply repair patterns
        for pattern, replacement in self.JSON_REPAIR_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.MULTILINE)

        # Try to parse after basic cleanup
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from mixed content
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try more aggressive repair
        return self._aggressive_repair(cleaned)

    def _aggressive_repair(self, content: str) -> dict[str, Any] | None:
        """More aggressive JSON repair attempts.

        Args:
            content: Content to repair.

        Returns:
            Parsed dict if successful, None otherwise.
        """
        # Try removing excessive whitespace
        try:
            compressed = re.sub(r"\s+", " ", content)
            return json.loads(compressed)
        except json.JSONDecodeError:
            pass

        # Try finding balanced braces
        try:
            start = content.index("{")
            depth = 0
            for i, char in enumerate(content[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(content[start : i + 1])
        except (ValueError, json.JSONDecodeError):
            pass

        # Try removing common prefixes/suffixes
        prefixes = [
            "Here is the JSON:",
            "Here's the JSON:",
            "JSON output:",
            "Output:",
            "Result:",
        ]
        for prefix in prefixes:
            if content.lower().startswith(prefix.lower()):
                trimmed = content[len(prefix) :].strip()
                try:
                    return json.loads(trimmed)
                except json.JSONDecodeError:
                    pass

        return None

    def get_backoff_delay(self, attempt: int) -> float:
        """Calculate backoff delay for retry attempt.

        Uses exponential backoff with configurable limits.

        Args:
            attempt: Current attempt number (0-based).

        Returns:
            Delay in seconds before next attempt.
        """
        if not self._config.exponential_backoff:
            return self._config.initial_delay

        delay = self._config.initial_delay * (2**attempt)
        return min(delay, self._config.max_delay)

    def should_retry(
        self,
        error: Exception,
        attempt: int,
    ) -> bool:
        """Determine if error should trigger retry.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (0-based).

        Returns:
            True if should retry, False otherwise.
        """
        if attempt >= self._config.max_retries:
            return False

        # Always retry parse errors (might be transient)
        if isinstance(error, json.JSONDecodeError):
            return True

        # Check error message for retryable conditions
        error_str = str(error).lower()
        retryable_patterns = [
            "rate limit",
            "rate_limit",
            "timeout",
            "connection",
            "temporarily unavailable",
            "overloaded",
            "503",
            "502",
            "500",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)


__all__ = ["RetryStrategy", "RetryConfig"]
