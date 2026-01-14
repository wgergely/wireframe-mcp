"""Feedback processing module for layout refinement.

Provides utilities for parsing natural language feedback
and generating modification hints for the generation pipeline.
"""

from src.feedback.lib import FeedbackIntent, FeedbackParser

__all__ = [
    "FeedbackIntent",
    "FeedbackParser",
]
