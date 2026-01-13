"""Enrico dataset provider."""

import json
import zipfile
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

from src.corpus.provider.base import BaseProvider, StandardizedData


class Provider(BaseProvider):
    """Provider for the Enrico dataset (subset of Rico with design topics)."""

    ENRICO_URL = (
        "https://huggingface.co/datasets/luileito/enrico/resolve/main/enrico.zip"
    )

    @property
    def name(self) -> str:
        """Provider name."""
        return "enrico"

    @property
    def _dest_dir(self) -> Path:
        """Base directory for Enrico data."""
        return self.data_dir / "enrico"

    @property
    def _extract_dir(self) -> Path:
        """Directory where the dataset is extracted."""
        return self._dest_dir / "extracted"

    def fetch(self, force: bool = False) -> None:
        """Download and extract Enrico dataset."""
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        if self._extract_dir.exists() and not force:
            print(f"[{self.name}] Dataset already exists at {self._extract_dir}")
            return

        zip_path = self._dest_dir / "enrico.zip"

        print(f"[{self.name}] Downloading from {self.ENRICO_URL}...")
        try:
            urlretrieve(self.ENRICO_URL, zip_path)
        except Exception as e:
            raise ConnectionError(f"[{self.name}] Download failed: {e}") from e

        print(f"[{self.name}] Extracting...")
        self._extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self._extract_dir)

        print(f"[{self.name}] Ready at {self._extract_dir}")

    def process(self) -> Iterator[StandardizedData]:
        """Process Enrico data and yield standardized items."""
        if not self._extract_dir.exists():
            raise FileNotFoundError(f"[{self.name}] Run fetch() first.")

        for json_path in self._extract_dir.rglob("*.json"):
            item = self._process_json_file(json_path)
            if item:
                yield item

    def _process_json_file(self, json_path: Path) -> StandardizedData | None:
        """Process a single JSON file and return StandardizedData."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Enrico often uses jpg for screenshots
            screenshot_path = json_path.with_suffix(".jpg")
            return StandardizedData(
                id=json_path.stem,
                source="enrico",
                dataset="default",
                hierarchy=data,
                metadata={"filename": json_path.name},
                screenshot_path=screenshot_path if screenshot_path.exists() else None,
            )
        except Exception as e:
            print(f"[{self.name}] Error reading {json_path}: {e}")
            return None
