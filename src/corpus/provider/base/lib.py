"""Base provider definitions for the Corpus module."""

import abc
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict

from src.mid import LayoutNode


class StandardizedData(BaseModel):
    """Represents a standardized unit of UI data from any provider.

    Attributes:
        id: Unique identifier for this data item.
        source: Name of the data source (e.g., 'rico', 'enrico').
        dataset: Dataset type within the source.
        hierarchy: The view hierarchy as raw JSON dict.
        layout: The view hierarchy as a LayoutNode (semantic MID).
        metadata: Additional metadata about the item.
        screenshot_path: Path to screenshot image, if available.
    """

    id: str
    source: str
    dataset: str
    hierarchy: dict
    layout: LayoutNode | None = None
    metadata: dict
    screenshot_path: Path | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseProvider(abc.ABC):
    """Abstract base class for corpus data providers.

    A provider is responsible for:
    1. Fetching raw data (downloading).
    2. Extracting raw data (unzipping/processing).
    3. Producing a stream of StandardizedData.

    Attributes:
        data_dir: Root directory for storing provider data.
    """

    def __init__(self, data_dir: Path):
        """Initialize the provider.

        Args:
            data_dir: Root directory where corpus data should be stored.
        """
        self.data_dir = data_dir

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider."""

    @abc.abstractmethod
    def fetch(self, force: bool = False) -> None:
        """Download or retrieve the raw data.

        Args:
            force: If True, force re-download even if data exists.
        """

    @abc.abstractmethod
    def process(self) -> Iterator[StandardizedData]:
        """Process raw data and yield standardized data items.

        This method reads the raw files (after fetch is called) and
        converts them into StandardizedData objects.

        Yields:
            StandardizedData items from the provider.
        """
