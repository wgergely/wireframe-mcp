"""Enrico dataset provider."""

import json
import zipfile
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
        return "enrico"

    def fetch(self, force: bool = False) -> None:
        """Download and extract Enrico dataset."""
        dest_dir = self.data_dir / "enrico"
        dest_dir.mkdir(parents=True, exist_ok=True)

        zip_path = dest_dir / "enrico.zip"
        extract_dir = dest_dir / "extracted"

        if extract_dir.exists() and not force:
            print(f"[{self.name}] Dataset already exists at {extract_dir}")
            return

        print(f"[{self.name}] Downloading from {self.ENRICO_URL}...")
        try:
            urlretrieve(self.ENRICO_URL, zip_path)
        except Exception as e:
            raise ConnectionError(f"[{self.name}] Download failed: {e}") from e

        print(f"[{self.name}] Extracting...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Clean up zip? Maybe keep it for cache.
        print(f"[{self.name}] Ready at {extract_dir}")

    def process(self) -> Iterator[StandardizedData]:
        """Process Enrico data."""
        extract_dir = self.data_dir / "enrico" / "extracted"
        if not extract_dir.exists():
            raise FileNotFoundError(f"[{self.name}] Run fetch() first.")

        # Structure of Enrico zip usually:
        # screenshots/
        # hierarchies/
        # ...
        # OR it might be flat.
        # I'll implement a resilient walker.

        for json_file in extract_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check if it's a hierarchy
                # Enrico/Rico usually has 'children' or 'activity_name'

                yield StandardizedData(
                    id=json_file.stem,
                    source="enrico",
                    dataset="default",
                    hierarchy=data,
                    metadata={"filename": json_file.name},
                    screenshot_path=json_file.with_suffix(
                        ".jpg"
                    ),  # Enrico often uses jpg
                )
            except Exception as e:
                print(f"[{self.name}] Error reading {json_file}: {e}")
