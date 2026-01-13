"""Rico dataset provider."""

import json
from pathlib import Path
from typing import Iterator, Literal

from src.core import get_logger
from src.corpus.normalizer import normalize_rico_hierarchy
from src.corpus.provider.base import (
    BaseProvider,
    DataType,
    StandardizedData,
    download_and_extract,
)
from src.mid import LayoutNode

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

    def has_data(self, data_type: DataType | None = None) -> bool:
        """Check if data exists.

        Args:
            data_type: Optional filter for specific data type.

        Returns:
            True if requested data is available, False otherwise.
        """
        if not self._extract_dir.exists():
            return False

        if data_type is None:
            return True
        if data_type == DataType.HIERARCHY:
            return any(self._extract_dir.rglob("*.json"))
        if data_type == DataType.IMAGE:
            return any(self._extract_dir.rglob("*.png"))
        if data_type in (DataType.LAYOUT, DataType.TEXT):
            return self.has_data(DataType.HIERARCHY)
        return False

    def to_layout(self, hierarchy: dict, item_id: str) -> "LayoutNode":
        """Convert Rico hierarchy to LayoutNode.

        Args:
            hierarchy: Rico view hierarchy dict.
            item_id: Unique identifier for generating node IDs.

        Returns:
            LayoutNode tree representing the semantic UI structure.
        """
        return normalize_rico_hierarchy(hierarchy, item_id)

    def fetch(self, force: bool = False) -> None:
        """Download and extract the Rico dataset.

        Args:
            force: If True, force re-download even if data exists.
        """
        if self.has_data() and not force:
            logger.info(f"[{self.name}] Dataset already exists at {self._extract_dir}")
            return

        url = self.dataset_info["url"]
        expected_size_mb = self.dataset_info.get("size_mb")
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        download_and_extract(
            url=url,
            dest_dir=self._dest_dir,
            extract_dir=self._extract_dir,
            provider_name=self.name,
            expected_size_mb=expected_size_mb,
        )
        logger.info(f"[{self.name}] Dataset ready at {self._extract_dir}")

    def process(self) -> Iterator[StandardizedData]:
        """Process Rico data and yield standardized items.

        Yields:
            StandardizedData items from the Rico dataset.

        Raises:
            FileNotFoundError: If dataset directory does not exist.
        """
        if not self.has_data():
            raise FileNotFoundError(
                f"[{self.name}] Dataset not found at {self._extract_dir}. "
                "Run fetch() first."
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
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] Skipping invalid JSON: {json_path}")
            return None
        except Exception as e:
            logger.error(f"[{self.name}] Error processing {json_path}: {e}")
            return None

        item_id = json_path.stem
        screenshot_path = json_path.with_suffix(".png")

        return StandardizedData(
            id=item_id,
            source="rico",
            dataset=self.dataset_type,
            hierarchy=data,
            layout=self.to_layout(data, item_id),
            metadata={"filename": json_path.name},
            screenshot_path=screenshot_path if screenshot_path.exists() else None,
        )
