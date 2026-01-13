"""Base provider definitions for the Corpus module."""

import abc
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict


class StandardizedData(BaseModel):
    """Represents a standardized unit of UI data from any provider."""

    id: str
    source: str
    dataset: str
    hierarchy: dict  # The view hierarchy (JSON)
    metadata: dict  # Additional metadata
    screenshot_path: Path | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseProvider(abc.ABC):
    """Abstract base class for corpus data providers.

    A provider is responsible for:
    1. Fetching raw data (downloading).
    2. Extracting raw data (unzipping/processing).
    3. PRODUCING a stream of StandardizedData.
    """

    def __init__(self, data_dir: Path):
        """Initialize the provider.

        Args:
            data_dir: The root directory where corpus data should be stored.
        """
        self.data_dir = data_dir

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider."""
        pass

    @abc.abstractmethod
    def fetch(self, force: bool = False) -> None:
        """Download or retrieve the raw data.

        Args:
            force: If True, force re-download even if data appears to exist.
        """
        pass

    @abc.abstractmethod
    def process(self) -> Iterator[StandardizedData]:
        """Process the raw data and yield standardized data items.

        This method should handle reading the raw files (after fetch is called)
        and converting them into StandardizedData objects.

        Yields:
            StandardizedData: The standardized data items.
        """
        pass
