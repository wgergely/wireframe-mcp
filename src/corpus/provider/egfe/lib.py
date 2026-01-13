"""EGFE (Expert Guided Figma Execution) dataset provider.

EGFE contains high-quality UI design prototypes with fragmented layered data.
Source: https://zenodo.org/records/8004165
"""

import json
import zipfile
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

from src.core import get_logger
from src.corpus.normalizer import hierarchy_to_layout
from src.corpus.provider.base import BaseProvider, StandardizedData
from src.mid import ComponentType

logger = get_logger("provider.egfe")

# EGFE dataset on Zenodo (partial release - 300 samples, MIT license)
EGFE_URL = "https://zenodo.org/records/8004165/files/EGFE-dataset.zip"

# Figma type to ComponentType mapping
FIGMA_TYPE_MAP: dict[str, ComponentType] = {
    "FRAME": ComponentType.CONTAINER,
    "RECTANGLE": ComponentType.CONTAINER,
    "GROUP": ComponentType.CONTAINER,
    "COMPONENT": ComponentType.CARD,
    "INSTANCE": ComponentType.CARD,
    "TEXT": ComponentType.TEXT,
    "VECTOR": ComponentType.ICON,
    "ELLIPSE": ComponentType.ICON,
}


def _infer_component_from_figma(node: dict) -> str:
    """Infer componentLabel from Figma node properties."""
    node_type = node.get("type", "FRAME")
    name = node.get("name", "").lower()

    # Check name patterns for common UI elements
    if "button" in name or "btn" in name:
        return "Text Button"
    if "input" in name or "field" in name:
        return "Input"
    if "icon" in name:
        return "Icon"
    if "image" in name or "img" in name:
        return "Image"
    if "card" in name:
        return "Card"
    if "nav" in name or "menu" in name or "header" in name:
        return "Toolbar"
    if "modal" in name or "dialog" in name:
        return "Modal"

    # Fall back to type mapping
    comp_type = FIGMA_TYPE_MAP.get(node_type, ComponentType.CONTAINER)
    return {
        ComponentType.CONTAINER: "Container",
        ComponentType.CARD: "Card",
        ComponentType.TEXT: "Text",
        ComponentType.ICON: "Icon",
    }.get(comp_type, "Container")


def _figma_to_hierarchy(node: dict) -> dict:
    """Convert Figma JSON node to Rico-compatible hierarchy format."""
    result = {
        "class": f"figma.{node.get('type', 'FRAME')}",
        "componentLabel": _infer_component_from_figma(node),
        "children": [],
    }

    # Extract bounds from absoluteBoundingBox or bounds
    bbox = node.get("absoluteBoundingBox") or node.get("bounds", {})
    if bbox:
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("width", 0))
        h = int(bbox.get("height", 0))
        result["bounds"] = [x, y, x + w, y + h]

    # Extract text content
    if node.get("type") == "TEXT":
        result["text"] = node.get("characters", node.get("name", ""))
    elif "name" in node:
        name = node["name"]
        if not name.startswith("Frame") and not name.startswith("Group"):
            result["text"] = name

    # Process children
    for child in node.get("children", []):
        result["children"].append(_figma_to_hierarchy(child))

    return result


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
            logger.info(f"[{self.name}] Extracting to {self._dest_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self._dest_dir)
            logger.info(f"[{self.name}] Ready at {self._dest_dir}")
        except Exception as e:
            logger.warning(f"[{self.name}] Download failed: {e}")
            logger.info("  Creating sample data for testing...")
            self._create_sample_data()

    def _create_sample_data(self) -> None:
        """Create sample Figma-like data for testing."""
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        sample = {
            "type": "FRAME",
            "name": "Mobile App Screen",
            "absoluteBoundingBox": {"x": 0, "y": 0, "width": 375, "height": 812},
            "children": [
                {
                    "type": "FRAME",
                    "name": "Header",
                    "absoluteBoundingBox": {"x": 0, "y": 0, "width": 375, "height": 60},
                    "children": [
                        {
                            "type": "TEXT",
                            "name": "Title",
                            "characters": "App Title",
                            "absoluteBoundingBox": {
                                "x": 16,
                                "y": 20,
                                "width": 200,
                                "height": 24,
                            },
                        }
                    ],
                },
                {
                    "type": "COMPONENT",
                    "name": "Card",
                    "absoluteBoundingBox": {
                        "x": 16,
                        "y": 80,
                        "width": 343,
                        "height": 150,
                    },
                    "children": [
                        {
                            "type": "TEXT",
                            "characters": "Welcome",
                            "absoluteBoundingBox": {
                                "x": 32,
                                "y": 96,
                                "width": 100,
                                "height": 24,
                            },
                        },
                        {
                            "type": "TEXT",
                            "characters": "Sample content",
                            "absoluteBoundingBox": {
                                "x": 32,
                                "y": 130,
                                "width": 200,
                                "height": 40,
                            },
                        },
                    ],
                },
                {
                    "type": "INSTANCE",
                    "name": "Primary Button",
                    "absoluteBoundingBox": {
                        "x": 16,
                        "y": 250,
                        "width": 343,
                        "height": 48,
                    },
                    "children": [
                        {
                            "type": "TEXT",
                            "characters": "Get Started",
                            "absoluteBoundingBox": {
                                "x": 140,
                                "y": 262,
                                "width": 95,
                                "height": 24,
                            },
                        }
                    ],
                },
            ],
        }

        with open(self._dest_dir / "sample_001.json", "w") as f:
            json.dump(sample, f, indent=2)
        logger.info(f"[{self.name}] Created sample data")

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

            # Convert Figma format to Rico-compatible hierarchy
            hierarchy = _figma_to_hierarchy(data)

            # Convert to LayoutNode
            item_id = json_path.stem
            layout = hierarchy_to_layout(hierarchy, id_prefix=f"egfe_{item_id}")

            screenshot_path = screenshot_lookup.get(json_path.stem)

            return StandardizedData(
                id=item_id,
                source="egfe",
                dataset="default",
                hierarchy=hierarchy,
                layout=layout,
                metadata={
                    "filename": json_path.name,
                    "figma_type": data.get("type", ""),
                    "figma_name": data.get("name", ""),
                },
                screenshot_path=screenshot_path,
            )
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] Skipping invalid JSON: {json_path}")
            return None
        except Exception as e:
            logger.error(f"[{self.name}] Error reading {json_path}: {e}")
            return None
