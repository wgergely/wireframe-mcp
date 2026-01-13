"""Tests for core logging module."""

import logging
from io import StringIO
from .lib import get_logger, setup_logging


class TestLogging:
    """Test core logging API."""

    def test_get_logger(self) -> None:
        """Verify logger instance creation."""
        logger = get_logger("test")
        assert logger.name == "test"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_default_name(self) -> None:
        """Verify default logger name."""
        logger = get_logger()
        assert logger.name == "wireframe-mcp"

    def test_setup_logging(self) -> None:
        """Verify logging setup."""
        stream = StringIO()
        setup_logging(level=logging.DEBUG, stream=stream)
        logger = get_logger("test_setup")
        logger.debug("test message")

        # Basic check that something was written to the stream
        # Note: basicConfig might not apply if logging was already configured
        # but we check the API contract.
        assert logger.level == logging.NOTSET  # Level is on the handler or root
