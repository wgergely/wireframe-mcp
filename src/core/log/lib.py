"""Core logging implementation for wireframe-mcp."""

import logging
import sys
from typing import Optional

__all__ = ["get_logger", "setup_logging"]


def setup_logging(level: int = logging.INFO, stream=sys.stderr) -> None:
    """Configure basic logging.

    Args:
        level: Logging level.
        stream: Output stream.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=stream,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name or "wireframe-mcp")
