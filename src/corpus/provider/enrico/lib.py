"""Enrico dataset provider."""

import json
from pathlib import Path
from typing import Iterator

from src.core import get_logger
from src.corpus.normalizer import normalize_enrico_hierarchy
from src.corpus.provider.base import (
    BaseProvider,
    DataType,
    StandardizedData,
    download_and_extract,
)
from src.mid import LayoutNode

logger = get_logger("provider.enrico")

# Enrico download URLs from userinterfaces.aalto.fi
ENRICO_URLS = {
    "hierarchies": "http://userinterfaces.aalto.fi/enrico/resources/hierarchies.zip",
    "screenshots": "http://userinterfaces.aalto.fi/enrico/resources/screenshots.zip",
}


class Provider(BaseProvider):
    """Provider for the Enrico dataset (subset of Rico with design topics)."""

    @property
    def name(self) -> str:
        """Provider name."""
        return "enrico"

    @property
    def _dest_dir(self) -> Path:
        """Base directory for Enrico data."""
        return self.data_dir / "enrico"

    @property
    def _hierarchies_dir(self) -> Path:
        """Directory containing view hierarchy JSON files."""
        return self._dest_dir / "hierarchies"

    @property
    def _screenshots_dir(self) -> Path:
        """Directory containing screenshot JPG files."""
        return self._dest_dir / "screenshots"

    def has_data(self, data_type: DataType | None = None) -> bool:
        """Check if data exists.

        Args:
            data_type: Optional filter for specific data type.

        Returns:
            True if requested data is available, False otherwise.
        """
        if data_type is None:
            return self._hierarchies_dir.exists() and self._screenshots_dir.exists()
        if data_type == DataType.HIERARCHY:
            return self._hierarchies_dir.exists() and any(
                self._hierarchies_dir.rglob("*.json")
            )
        if data_type == DataType.IMAGE:
            return self._screenshots_dir.exists() and any(
                self._screenshots_dir.rglob("*.jpg")
            )
        if data_type in (DataType.LAYOUT, DataType.TEXT):
            return self.has_data(DataType.HIERARCHY)
        return False

    def to_layout(self, hierarchy: dict, item_id: str) -> "LayoutNode":
        """Convert Enrico hierarchy to LayoutNode.

        Args:
            hierarchy: Enrico view hierarchy dict.
            item_id: Unique identifier for generating node IDs.

        Returns:
            LayoutNode tree representing the semantic UI structure.
        """
        return normalize_enrico_hierarchy(hierarchy, item_id)

    def fetch(self, force: bool = False) -> None:
        """Download and extract Enrico dataset (hierarchies + screenshots).

        Args:
            force: If True, force re-download even if data exists.
        """
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if self.has_data() and not force:
            logger.info(f"[{self.name}] Dataset already exists at {self._dest_dir}")
            return

        # Download and extract hierarchies (~5MB)
        logger.info(f"[{self.name}] Downloading hierarchies...")
        download_and_extract(
            url=ENRICO_URLS["hierarchies"],
            dest_dir=self._dest_dir,
            extract_dir=self._hierarchies_dir,
            provider_name=self.name,
            expected_size_mb=5,
        )

        # Download and extract screenshots (~200MB)
        logger.info(f"[{self.name}] Downloading screenshots...")
        download_and_extract(
            url=ENRICO_URLS["screenshots"],
            dest_dir=self._dest_dir,
            extract_dir=self._screenshots_dir,
            provider_name=self.name,
            expected_size_mb=200,
        )

        logger.info(f"[{self.name}] Ready at {self._dest_dir}")

    def process(self) -> Iterator[StandardizedData]:
        """Process Enrico data and yield standardized items.

        Yields:
            StandardizedData items from the Enrico dataset.

        Raises:
            FileNotFoundError: If dataset directory does not exist.
        """
        if not self.has_data():
            raise FileNotFoundError(f"[{self.name}] Run fetch() first.")

        # Build screenshot lookup once for efficiency
        screenshot_lookup = {p.stem: p for p in self._screenshots_dir.rglob("*.jpg")}

        for json_path in self._hierarchies_dir.rglob("*.json"):
            item = self._process_json_file(json_path, screenshot_lookup)
            if item:
                yield item

    def _process_json_file(
        self, json_path: Path, screenshot_lookup: dict[str, Path]
    ) -> StandardizedData | None:
        """Process a single JSON file and return StandardizedData.

        Args:
            json_path: Path to the JSON file.
            screenshot_lookup: Dict mapping file stems to screenshot paths.

        Returns:
            StandardizedData if successful, None if parsing fails.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] Skipping invalid JSON: {json_path}")
            return None
        except Exception as e:
            logger.error(f"[{self.name}] Error reading {json_path}: {e}")
            return None

        item_id = json_path.stem

        return StandardizedData(
            id=item_id,
            source="enrico",
            dataset="default",
            hierarchy=data,
            layout=self.to_layout(data, item_id),
            metadata={"filename": json_path.name},
            screenshot_path=screenshot_lookup.get(item_id),
        )
