"""Storage backends for history management.

This module provides storage implementations for persisting
generation artifacts, sessions, and variation sets.

Available backends:
- SQLiteStorage: File-based SQLite database (recommended for local use)
- InMemoryStorage: In-memory storage for testing
"""

from .protocol import HistoryStorage
from .sqlite import SQLiteStorage

__all__ = [
    "HistoryStorage",
    "SQLiteStorage",
]
