"""Tests for feedback module."""

import pytest

from src.feedback import FeedbackIntent, FeedbackParser
from src.feedback.lib import FeedbackAction


class TestFeedbackParser:
    """Tests for FeedbackParser class."""

    @pytest.fixture
    def parser(self):
        return FeedbackParser()

    @pytest.mark.unit
    def test_parse_add(self, parser):
        """Test parsing add intents."""
        intents = parser.parse("Add a search bar to the header")
        assert len(intents) == 1
        assert intents[0].action == FeedbackAction.ADD
        assert "search bar" in intents[0].target

    @pytest.mark.unit
    def test_parse_remove(self, parser):
        """Test parsing remove intents."""
        intents = parser.parse("Remove the advertisement section")
        assert len(intents) == 1
        assert intents[0].action == FeedbackAction.REMOVE

    @pytest.mark.unit
    def test_parse_resize(self, parser):
        """Test parsing resize intents."""
        intents = parser.parse("Make the sidebar narrower")
        assert len(intents) == 1
        assert intents[0].action == FeedbackAction.RESIZE

    @pytest.mark.unit
    def test_parse_modify(self, parser):
        """Test parsing modify intents."""
        intents = parser.parse("Change the button color to blue")
        assert len(intents) == 1
        assert intents[0].action == FeedbackAction.MODIFY

    @pytest.mark.unit
    def test_parse_move(self, parser):
        """Test parsing move intents."""
        intents = parser.parse("Move the logo to the left")
        assert len(intents) == 1
        assert intents[0].action == FeedbackAction.MOVE

    @pytest.mark.unit
    def test_fallback_to_modify(self, parser):
        """Test fallback for unrecognized patterns."""
        intents = parser.parse("I want something different")
        assert len(intents) == 1
        assert intents[0].action == FeedbackAction.MODIFY
        assert intents[0].confidence < 0.5

    @pytest.mark.unit
    def test_empty_input(self, parser):
        """Test empty input handling."""
        intents = parser.parse("")
        assert len(intents) == 0


class TestFeedbackIntent:
    """Tests for FeedbackIntent dataclass."""

    @pytest.mark.unit
    def test_intent_creation(self):
        """Test creating a FeedbackIntent."""
        intent = FeedbackIntent(
            action=FeedbackAction.ADD,
            target="button",
            details="add a submit button",
            confidence=0.9,
        )
        assert intent.action == FeedbackAction.ADD
        assert intent.target == "button"
        assert intent.confidence == 0.9
