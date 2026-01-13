"""ShowUI Desktop dataset provider."""

import json
from pathlib import Path
from typing import Iterator

from src.core import get_logger
from src.corpus.provider.base import BaseProvider, DataType, StandardizedData
from src.mid import ComponentType, LayoutNode

logger = get_logger("provider.showui")

# Default screen dimensions for converting normalized coords
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080

# ShowUI Desktop dataset on HuggingFace
SHOWUI_DATASET = "Voxel51/ShowUI_desktop"


class Provider(BaseProvider):
    """Provider for the ShowUI Desktop dataset.

    ShowUI contains desktop GUI element detections with bounding boxes
    and grounding instructions. Unlike Rico/Enrico, it has a flat structure
    (no tree hierarchy) - elements are independent detections.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "showui"

    @property
    def _dest_dir(self) -> Path:
        """Base directory for ShowUI data."""
        return self.data_dir / "showui"

    @property
    def _samples_dir(self) -> Path:
        """Directory containing processed samples."""
        return self._dest_dir / "samples"

    def has_data(self, data_type: DataType | None = None) -> bool:
        """Check if data exists (JSON samples or HuggingFace cache).

        Args:
            data_type: Optional filter for specific data type.

        Returns:
            True if requested data is available, False otherwise.
        """
        if not self._samples_dir.exists():
            return False

        if data_type == DataType.IMAGE:
            return False  # ShowUI samples don't include screenshot files
        if data_type is None or data_type in (
            DataType.HIERARCHY,
            DataType.LAYOUT,
            DataType.TEXT,
        ):
            return any(self._samples_dir.glob("*.json"))
        return False

    def to_layout(self, hierarchy: dict, item_id: str) -> LayoutNode:
        """Convert provider-specific hierarchy to LayoutNode.

        Args:
            hierarchy: ShowUI pseudo-hierarchy dict.
            item_id: Unique identifier for generating node IDs.

        Returns:
            LayoutNode tree representing the semantic UI structure.
        """
        return self._showui_hierarchy_to_layout(
            hierarchy, id_prefix=f"showui_{item_id}"
        )

    def fetch(self, force: bool = False) -> None:
        """Download ShowUI Desktop dataset.

        This provider supports two modes:
        1. If `datasets` library is available: Download from HuggingFace
        2. Otherwise: Provide instructions for manual download

        Args:
            force: If True, force re-download even if data exists.
        """
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        if self.has_data() and not force:
            logger.info(f"[{self.name}] Dataset already exists at {self._dest_dir}")
            return

        # Try to use HuggingFace datasets library
        try:
            from datasets import load_dataset

            logger.info(f"[{self.name}] Downloading from HuggingFace: {SHOWUI_DATASET}")
            dataset = load_dataset(SHOWUI_DATASET, split="train")

            # Save samples as JSON files for offline processing
            self._samples_dir.mkdir(parents=True, exist_ok=True)

            for i, sample in enumerate(dataset):
                sample_path = self._samples_dir / f"sample_{i:05d}.json"
                # Convert to serializable format
                serializable = self._sample_to_dict(sample)
                with open(sample_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f)

            logger.info(
                f"[{self.name}] Saved {len(dataset)} samples to {self._samples_dir}"
            )

        except ImportError:
            logger.warning(
                f"[{self.name}] 'datasets' library not installed. "
                "Install with: pip install datasets"
            )
            logger.info(f"[{self.name}] Creating sample data for testing...")
            self._create_sample_data()

    def _create_sample_data(self) -> None:
        """Create sample data for testing without HuggingFace."""
        self._samples_dir.mkdir(parents=True, exist_ok=True)

        samples = [
            {
                "instruction": "Click the Settings button",
                "detections": [
                    {"label": "button", "bounding_box": [0.85, 0.02, 0.1, 0.05]},
                ],
                "query_type": "click",
                "interfaces": "desktop",
            },
            {
                "instruction": "Type in the search box",
                "detections": [
                    {"label": "input", "bounding_box": [0.3, 0.1, 0.4, 0.05]},
                ],
                "query_type": "type",
                "interfaces": "desktop",
            },
            {
                "instruction": "Select the file icon",
                "detections": [
                    {"label": "icon", "bounding_box": [0.05, 0.2, 0.08, 0.1]},
                    {"label": "icon", "bounding_box": [0.15, 0.2, 0.08, 0.1]},
                    {"label": "icon", "bounding_box": [0.25, 0.2, 0.08, 0.1]},
                ],
                "query_type": "click",
                "interfaces": "desktop",
            },
        ]

        for i, sample in enumerate(samples):
            sample_path = self._samples_dir / f"sample_{i:05d}.json"
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(sample, f)

        logger.info(f"[{self.name}] Created {len(samples)} sample items")

    def _sample_to_dict(self, sample: dict) -> dict:
        """Convert HuggingFace sample to serializable dict.

        Args:
            sample: Raw sample from HuggingFace dataset.

        Returns:
            Serializable dictionary without PIL images.
        """
        result = {}

        # Copy text fields
        if "instruction" in sample:
            result["instruction"] = sample["instruction"]

        # Convert detections (bounding boxes)
        if "action_detections" in sample:
            detections = sample["action_detections"]
            if hasattr(detections, "detections"):
                result["detections"] = [
                    {
                        "label": d.get("label", "action"),
                        "bounding_box": list(d.get("bounding_box", [])),
                    }
                    for d in detections.detections
                ]
            elif isinstance(detections, dict):
                result["detections"] = detections.get("detections", [])

        # Convert keypoints
        if "action_keypoints" in sample:
            keypoints = sample["action_keypoints"]
            if hasattr(keypoints, "keypoints"):
                result["keypoints"] = [
                    {
                        "label": k.get("label", "action"),
                        "points": list(k.get("points", [])),
                    }
                    for k in keypoints.keypoints
                ]
            elif isinstance(keypoints, dict):
                result["keypoints"] = keypoints.get("keypoints", [])

        # Copy metadata
        if "query_type" in sample:
            qt = sample["query_type"]
            result["query_type"] = qt.get("label") if isinstance(qt, dict) else str(qt)

        if "interfaces" in sample:
            iface = sample["interfaces"]
            result["interfaces"] = (
                iface.get("label") if isinstance(iface, dict) else str(iface)
            )

        return result

    def process(self) -> Iterator[StandardizedData]:
        """Process ShowUI data and yield standardized items.

        Converts flat detections to pseudo-hierarchy with depth=1.

        Yields:
            StandardizedData items from the ShowUI dataset.
        """
        if not self.has_data():
            raise FileNotFoundError(f"[{self.name}] Run fetch() first.")

        for json_path in sorted(self._samples_dir.glob("*.json")):
            item = self._process_json_file(json_path)
            if item:
                yield item

    def _showui_hierarchy_to_layout(
        self,
        hierarchy: dict,
        id_prefix: str = "showui",
    ) -> LayoutNode:
        """Convert ShowUI pseudo-hierarchy to LayoutNode.

        ShowUI uses normalized coordinates [x, y, w, h] in 0-1 range.
        Convert these to flex ratios for the LayoutNode structure.

        Args:
            hierarchy: ShowUI pseudo-hierarchy dict.
            id_prefix: Prefix for node IDs.

        Returns:
            Root LayoutNode with children.
        """
        children = []

        for i, child in enumerate(hierarchy.get("children", [])):
            bbox = child.get("bounds", [0, 0, 0.1, 0.1])
            width_ratio = bbox[2] if len(bbox) >= 4 else 0.1
            flex_ratio = max(1, min(12, round(width_ratio * 12)))

            comp_type = self._infer_component_type(child.get("label", "action"))

            children.append(
                LayoutNode(
                    id=f"{id_prefix}_{i}",
                    type=comp_type,
                    label=child.get("text") or None,
                    flex_ratio=flex_ratio,
                    children=[],
                )
            )

        return LayoutNode(
            id=f"{id_prefix}_root",
            type=ComponentType.CONTAINER,
            label=hierarchy.get("text") or None,
            flex_ratio=12,
            children=children,
        )

    def _infer_component_type(self, label: str) -> ComponentType:
        """Infer ComponentType from ShowUI label.

        Args:
            label: The label text from ShowUI detection.

        Returns:
            Appropriate ComponentType for the label.
        """
        label_lower = label.lower()
        if "button" in label_lower or "click" in label_lower:
            return ComponentType.BUTTON
        if "text" in label_lower or "input" in label_lower:
            return ComponentType.INPUT
        if "icon" in label_lower:
            return ComponentType.ICON
        if "image" in label_lower:
            return ComponentType.IMAGE
        return ComponentType.CONTAINER

    def _process_json_file(self, json_path: Path) -> StandardizedData | None:
        """Process a single JSON sample and return StandardizedData.

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
            logger.error(f"[{self.name}] Error reading {json_path}: {e}")
            return None

        item_id = json_path.stem
        hierarchy = self._detection_to_hierarchy(data)

        return StandardizedData(
            id=item_id,
            source="showui",
            dataset="desktop",
            hierarchy=hierarchy,
            layout=self.to_layout(hierarchy, item_id),
            metadata={
                "filename": json_path.name,
                "query_type": data.get("query_type"),
                "interfaces": data.get("interfaces"),
            },
            screenshot_path=None,
        )

    def _detection_to_hierarchy(self, data: dict) -> dict:
        """Convert flat detections to pseudo-hierarchy.

        Creates a root "screen" container with flat children for each detection.

        Args:
            data: Sample data with detections and instruction.

        Returns:
            Pseudo-hierarchical structure with depth=1.
        """
        instruction = data.get("instruction", "")
        detections = data.get("detections", [])

        children = []
        for i, det in enumerate(detections):
            bbox = det.get("bounding_box", [0, 0, 0, 0])
            children.append(
                {
                    "type": "element",
                    "id": f"element_{i}",
                    "bounds": bbox,  # Normalized [x, y, w, h] format
                    "text": instruction,
                    "label": det.get("label", "action"),
                }
            )

        return {
            "type": "screen",
            "id": "root",
            "bounds": [0, 0, 1, 1],  # Normalized full screen
            "text": instruction,
            "children": children,
        }
