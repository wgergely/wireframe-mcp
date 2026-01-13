"""Enrico dataset provider."""

import json
import zipfile
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

from src.core import get_logger
from src.corpus.normalizer import normalize_enrico_hierarchy
from src.corpus.provider.base import BaseProvider, StandardizedData

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

    def fetch(self, force: bool = False) -> None:
        """Download and extract Enrico dataset (hierarchies + screenshots)."""
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if self._hierarchies_dir.exists() and self._screenshots_dir.exists():
            if not force:
                logger.info(f"[{self.name}] Dataset already exists at {self._dest_dir}")
                return

        # Download and extract hierarchies
        self._download_and_extract(
            "hierarchies",
            ENRICO_URLS["hierarchies"],
            self._hierarchies_dir,
        )

        # Download and extract screenshots
        self._download_and_extract(
            "screenshots",
            ENRICO_URLS["screenshots"],
            self._screenshots_dir,
        )

        logger.info(f"[{self.name}] Ready at {self._dest_dir}")

    def _download_and_extract(self, name: str, url: str, extract_dir: Path) -> None:
        """Download and extract a single archive."""
        zip_path = self._dest_dir / f"{name}.zip"

        logger.info(f"[{self.name}] Downloading {name} from {url}...")
        try:
            urlretrieve(url, zip_path)
        except Exception as e:
            raise ConnectionError(f"[{self.name}] Download failed: {e}") from e

        logger.info(f"[{self.name}] Extracting {name}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    def process(self) -> Iterator[StandardizedData]:
        """Process Enrico data and yield standardized items."""
        if not self._hierarchies_dir.exists():
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
        """Process a single JSON file and return StandardizedData."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            screenshot_path = screenshot_lookup.get(json_path.stem)

            return StandardizedData(
                id=json_path.stem,
                source="enrico",
                dataset="default",
                hierarchy=data,
                metadata={"filename": json_path.name},
                screenshot_path=screenshot_path,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Error reading {json_path}: {e}")
            return None
