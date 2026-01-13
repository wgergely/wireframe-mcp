"""Rico dataset provider."""

import json
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, Literal
from urllib.request import urlretrieve

from src.corpus.provider.base import BaseProvider, StandardizedData

# Rico dataset URLs
RICO_DATASETS = {
    "semantic": {
        "url": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip",
        "size_mb": 150,
        "description": "UI screenshots and hierarchies with semantic annotations",
    },
    "ui_screenshots": {
        "url": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz",
        "size_mb": 6000,
        "description": "66k+ unique UI screenshots and view hierarchies",
    },
    "ui_metadata": {
        "url": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv",
        "size_mb": 2,
        "description": "Metadata about each UI screen",
    },
    "ui_vectors": {
        "url": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_layout_vectors.zip",
        "size_mb": 8,
        "description": "64-dimensional layout vectors for each UI",
    },
}

RicoDatasetType = Literal["semantic", "ui_screenshots", "ui_metadata", "ui_vectors"]


class Provider(BaseProvider):
    """Provider for the Rico UI dataset."""

    def __init__(self, data_dir: Path, dataset_type: RicoDatasetType = "semantic"):
        super().__init__(data_dir)
        self.dataset_type = dataset_type
        self.dataset_info = RICO_DATASETS[dataset_type]

    @property
    def name(self) -> str:
        return f"rico_{self.dataset_type}"

    @property
    def _base_dir(self) -> Path:
        """Directory for Rico content within the data directory."""
        return self.data_dir / "rico"

    @property
    def _extract_dir(self) -> Path:
        """Directory where the dataset is extracted."""
        return self._base_dir / self.dataset_type

    def fetch(self, force: bool = False) -> None:
        """Download and extract the Rico dataset."""
        if self._extract_dir.exists() and not force:
            print(f"Dataset already exists at {self._extract_dir}")
            return

        url = self.dataset_info["url"]
        self._base_dir.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        output_path = self._base_dir / filename

        print(f"Downloading Rico {self.dataset_type} dataset from {url}...")

        # Simple download with progress
        def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

        try:
            urlretrieve(url, output_path, reporthook=_progress_hook)
            print()
        except Exception as e:
            raise ConnectionError(f"Failed to download {url}: {e}") from e

        print(f"Extracting to {self._extract_dir}...")
        self._extract_dir.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".gz" or output_path.name.endswith(".tar.gz"):
            with tarfile.open(output_path, "r:gz") as tar:
                tar.extractall(path=self._extract_dir)
        elif output_path.suffix == ".zip":
            with zipfile.ZipFile(output_path, "r") as zf:
                zf.extractall(path=self._extract_dir)

        print(f"Dataset ready at {self._extract_dir}")

    def process(self) -> Iterator[StandardizedData]:
        """Process Rico data (starting with Semantic)."""
        if not self._extract_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._extract_dir}. Run fetch() first."
            )

        # Logic depends on dataset type.
        # For 'semantic', it's a zip containing JSONs (and pngs, structure varies).
        # Assuming the semantic zip extracts a folder which contains the JSONs.

        # We need to explore the extracted directory to find the files.
        # Based on previous knowledge, 'semantic_annotations.zip' likely contains a folder.

        # Let's try to walk the directory
        for file_path in self._extract_dir.rglob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Rico Semantic JSONs usually have "children" or "hierarchy"
                # They are file-per-screen.
                file_id = file_path.stem

                # We need to constructing StandardizedData
                # Rico semantic JSON IS the view hierarchy.

                screenshot_path = file_path.with_suffix(".png")
                yield StandardizedData(
                    id=file_id,
                    source="rico",
                    dataset=self.dataset_type,
                    hierarchy=data,
                    metadata={"filename": file_path.name},
                    screenshot_path=(
                        screenshot_path if screenshot_path.exists() else None
                    ),
                )
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
