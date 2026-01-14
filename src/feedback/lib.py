"""Feedback parsing for layout refinement.

Extracts structured modification intents from natural language feedback.
"""

from dataclasses import dataclass
from enum import Enum


class FeedbackAction(Enum):
    """Types of layout modifications."""

    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"
    MOVE = "move"
    RESIZE = "resize"


@dataclass
class FeedbackIntent:
    """Parsed user feedback intent.

    Attributes:
        action: Type of modification.
        target: Natural language description of target element.
        details: Modification details.
        confidence: Parse confidence 0.0-1.0.
    """

    action: FeedbackAction
    target: str
    details: str
    confidence: float = 0.5


class FeedbackParser:
    """Parse natural language corrections into structured intents.

    Uses pattern matching to extract modification requests from
    user feedback text.

    Example:
        >>> parser = FeedbackParser()
        >>> intents = parser.parse("Add a search bar to the header")
        >>> print(intents[0].action)
        FeedbackAction.ADD
    """

    # Pattern keywords for each action
    _ADD_KEYWORDS = ["add", "include", "insert", "put", "create"]
    _REMOVE_KEYWORDS = ["remove", "delete", "hide", "drop", "get rid of"]
    _MODIFY_KEYWORDS = ["change", "make", "update", "set", "modify"]
    _MOVE_KEYWORDS = ["move", "relocate", "put", "place"]
    _RESIZE_KEYWORDS = ["wider", "narrower", "bigger", "smaller", "resize"]

    def parse(self, feedback: str) -> list[FeedbackIntent]:
        """Parse feedback into structured intents.

        Args:
            feedback: User's natural language feedback.

        Returns:
            List of parsed FeedbackIntent objects.
        """
        intents: list[FeedbackIntent] = []
        feedback_lower = feedback.lower()

        # Check for resize patterns
        for keyword in self._RESIZE_KEYWORDS:
            if keyword in feedback_lower:
                intents.append(
                    FeedbackIntent(
                        action=FeedbackAction.RESIZE,
                        target=self._extract_target(feedback, keyword),
                        details=feedback,
                        confidence=0.7,
                    )
                )
                return intents

        # Check for add patterns
        for keyword in self._ADD_KEYWORDS:
            if keyword in feedback_lower:
                intents.append(
                    FeedbackIntent(
                        action=FeedbackAction.ADD,
                        target=self._extract_target(feedback, keyword),
                        details=feedback,
                        confidence=0.8,
                    )
                )
                return intents

        # Check for remove patterns
        for keyword in self._REMOVE_KEYWORDS:
            if keyword in feedback_lower:
                intents.append(
                    FeedbackIntent(
                        action=FeedbackAction.REMOVE,
                        target=self._extract_target(feedback, keyword),
                        details=feedback,
                        confidence=0.8,
                    )
                )
                return intents

        # Check for move patterns
        for keyword in self._MOVE_KEYWORDS:
            if keyword in feedback_lower:
                intents.append(
                    FeedbackIntent(
                        action=FeedbackAction.MOVE,
                        target=self._extract_target(feedback, keyword),
                        details=feedback,
                        confidence=0.7,
                    )
                )
                return intents

        # Check for modify patterns
        for keyword in self._MODIFY_KEYWORDS:
            if keyword in feedback_lower:
                intents.append(
                    FeedbackIntent(
                        action=FeedbackAction.MODIFY,
                        target=self._extract_target(feedback, keyword),
                        details=feedback,
                        confidence=0.6,
                    )
                )
                return intents

        # Fallback: general modification
        if feedback.strip():
            intents.append(
                FeedbackIntent(
                    action=FeedbackAction.MODIFY,
                    target="layout",
                    details=feedback,
                    confidence=0.3,
                )
            )

        return intents

    def _extract_target(self, feedback: str, keyword: str) -> str:
        """Extract target element from feedback after keyword."""
        lower = feedback.lower()
        idx = lower.find(keyword)
        if idx == -1:
            return "element"

        # Get text after keyword
        after = feedback[idx + len(keyword) :].strip()

        # Take first noun phrase (simplified)
        words = after.split()
        if not words:
            return "element"

        # Take up to 4 words as target
        target_words = []
        for word in words[:4]:
            if word.lower() in ["to", "from", "in", "the", "a", "an"]:
                continue
            target_words.append(word)

        return " ".join(target_words) or "element"


__all__ = [
    "FeedbackAction",
    "FeedbackIntent",
    "FeedbackParser",
]
