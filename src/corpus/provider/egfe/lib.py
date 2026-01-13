"""EGFE (Expert Guided Figma Execution) dataset provider."""

import json
import zipfile
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

from src.core import get_logger
from src.corpus.provider.base import BaseProvider, StandardizedData

logger = get_logger("provider.egfe")

# EGFE dataset on Zenodo (partial release - 300 samples, MIT license)
# Note: Full dataset requires contacting authors
EGFE_URL = "https://zenodo.org/records/10048053/files/EGFE_partial.zip"


class Provider(BaseProvider):
    """Provider for the EGFE dataset (Expert Guided Figma Execution).

    EGFE contains high-fidelity UI design prototypes from Sketch/Figma with
    layered data, including screenshots and JSON metadata.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "egfe"

    @property
    def _dest_dir(self) -> Path:
        """Base directory for EGFE data."""
        return self.data_dir / "egfe"

    @property
    def _json_dir(self) -> Path:
        """Directory containing UI element JSON files."""
        return self._dest_dir / "json"

    @property
    def _screenshots_dir(self) -> Path:
        """Directory containing UI screenshot PNG files."""
        return self._dest_dir / "screenshots"

    def fetch(self, force: bool = False) -> None:
        """Download and extract EGFE dataset from Zenodo.

        Args:
            force: If True, force re-download even if data exists.
        """
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if self._has_data() and not force:
            logger.info(f"[{self.name}] Dataset already exists at {self._dest_dir}")
            return

        zip_path = self._dest_dir / "egfe.zip"

        logger.info(f"[{self.name}] Downloading from {EGFE_URL}...")
        try:
            urlretrieve(EGFE_URL, zip_path)
        except Exception as e:
            raise ConnectionError(f"[{self.name}] Download failed: {e}") from e

        logger.info(f"[{self.name}] Extracting to {self._dest_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self._dest_dir)

        logger.info(f"[{self.name}] Ready at {self._dest_dir}")

    def _has_data(self) -> bool:
        """Check if data exists (either flat structure or subdirs)."""
        # Check for JSON files in any structure
        json_files = list(self._dest_dir.rglob("*.json"))
        return len(json_files) > 0

    def process(self) -> Iterator[StandardizedData]:
        """Process EGFE data and yield standardized items.

        Yields:
            StandardizedData items from the EGFE dataset.
        """
        if not self._has_data():
            raise FileNotFoundError(f"[{self.name}] Run fetch() first.")

        # Build screenshot lookup (handle both flat and nested structures)
        screenshot_lookup: dict[str, Path] = {}
        for ext in ("*.png", "*.jpg"):
            for p in self._dest_dir.rglob(ext):
                # Skip asset files (xxx-assets.png)
                if "-assets" not in p.stem:
                    screenshot_lookup[p.stem] = p

        # Process all JSON files
        for json_path in self._dest_dir.rglob("*.json"):
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

            screenshot_path = screenshot_lookup.get(json_path.stem)

            return StandardizedData(
                id=json_path.stem,
                source="egfe",
                dataset="default",
                hierarchy=data,
                metadata={"filename": json_path.name},
                screenshot_path=screenshot_path,
            )
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] Skipping invalid JSON: {json_path}")
            return None
        except Exception as e:
            logger.error(f"[{self.name}] Error reading {json_path}: {e}")
            return None
