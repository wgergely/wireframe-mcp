"""Rico dataset provider."""

import json
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, Literal
from urllib.request import urlretrieve

from src.core import get_logger
from src.corpus.normalizer import normalize_rico_hierarchy
from src.corpus.provider.base import BaseProvider, StandardizedData

logger = get_logger("provider.rico")

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
        """Initialize the Rico provider.

        Args:
            data_dir: Root directory for storing data.
            dataset_type: Which Rico dataset variant to use.
        """
        super().__init__(data_dir)
        self.dataset_type = dataset_type
        self.dataset_info = RICO_DATASETS[dataset_type]

    @property
    def name(self) -> str:
        """Provider name including dataset type."""
        return f"rico_{self.dataset_type}"

    @property
    def _dest_dir(self) -> Path:
        """Base directory for Rico data."""
        return self.data_dir / "rico"

    @property
    def _extract_dir(self) -> Path:
        """Directory where the dataset is extracted."""
        return self._dest_dir / self.dataset_type

    def _has_data(self) -> bool:
        """Check if data exists.

        Returns:
            True if the extracted dataset directory exists, False otherwise.
        """
        return self._extract_dir.exists()

    def fetch(self, force: bool = False) -> None:
        """Download and extract the Rico dataset.

        Args:
            force: If True, force re-download even if data exists.
        """
        if self._has_data() and not force:
            logger.info(f"[{self.name}] Dataset already exists at {self._extract_dir}")
            return

        url = self.dataset_info["url"]
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        output_path = self._dest_dir / filename

        logger.info(f"[{self.name}] Downloading from {url}...")
        self._download_with_progress(url, output_path)

        logger.info(f"[{self.name}] Extracting to {self._extract_dir}...")
        self._extract_archive(output_path)
        logger.info(f"[{self.name}] Dataset ready at {self._extract_dir}")

    def _download_with_progress(self, url: str, output_path: Path) -> None:
        """Download file with progress reporting.

        Args:
            url: URL to download from.
            output_path: Path to save the downloaded file.

        Raises:
            ConnectionError: If download fails.
        """

        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100, (block_num * block_size / total_size) * 100)
                logger.debug(f"[{self.name}] Progress: {percent:.1f}%")

        try:
            urlretrieve(url, output_path, reporthook=progress_hook)
        except Exception as e:
            raise ConnectionError(f"[{self.name}] Failed to download: {e}") from e

    def _extract_archive(self, archive_path: Path) -> None:
        """Extract archive to the extract directory.

        Args:
            archive_path: Path to the archive file (.tar.gz or .zip).
        """
        self._extract_dir.mkdir(parents=True, exist_ok=True)

        if archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self._extract_dir)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(path=self._extract_dir)

    def process(self) -> Iterator[StandardizedData]:
        """Process Rico data and yield standardized items.

        Yields:
            StandardizedData items from the Rico dataset.

        Raises:
            FileNotFoundError: If dataset directory does not exist.
        """
        if not self._has_data():
            raise FileNotFoundError(
                f"[{self.name}] Dataset not found at {self._extract_dir}. Run fetch() first."
            )

        for json_path in self._extract_dir.rglob("*.json"):
            item = self._process_json_file(json_path)
            if item:
                yield item

    def _process_json_file(self, json_path: Path) -> StandardizedData | None:
        """Process a single JSON file and return StandardizedData.

        Args:
            json_path: Path to the JSON file.

        Returns:
            StandardizedData if successful, None if parsing fails.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            screenshot_path = json_path.with_suffix(".png")
            item_id = json_path.stem

            # Normalize hierarchy to LayoutNode
            layout = normalize_rico_hierarchy(data, item_id)

            return StandardizedData(
                id=item_id,
                source="rico",
                dataset=self.dataset_type,
                hierarchy=data,
                layout=layout,
                metadata={"filename": json_path.name},
                screenshot_path=screenshot_path if screenshot_path.exists() else None,
            )
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] Skipping invalid JSON: {json_path}")
            return None
        except Exception as e:
            logger.error(f"[{self.name}] Error processing {json_path}: {e}")
            return None
