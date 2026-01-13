"""Corpus data fetching and management.

This module provides APIs for downloading external UI datasets (Rico, etc.)
to a configurable data directory. Designed for containerization scenarios
where data needs to be fetched programmatically.
"""

import tarfile
from pathlib import Path
from typing import Callable, Literal
from urllib.request import urlretrieve

from pydantic import BaseModel

# Rico dataset URLs from interactionmining.org
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

# Default data directory relative to project root
DEFAULT_DATA_DIR = Path("data")


class CorpusDataset(BaseModel):
    """Represents a downloaded corpus dataset.

    Attributes:
        name: Dataset identifier (e.g., "rico_semantic").
        path: Absolute path to the dataset directory.
        source: Source identifier (e.g., "rico").
        dataset_type: Specific dataset type within the source.
        downloaded: Whether the dataset has been fully downloaded.
    """

    name: str
    path: Path
    source: str
    dataset_type: str
    downloaded: bool = False

    model_config = {"arbitrary_types_allowed": True}


def get_data_dir(base_dir: Path | str | None = None) -> Path:
    """Get the data directory path, creating it if necessary.

    Args:
        base_dir: Optional base directory. If None, uses DEFAULT_DATA_DIR
                  relative to the current working directory.

    Returns:
        Path: Absolute path to the data directory.

    Example:
        >>> data_dir = get_data_dir()
        >>> data_dir.exists()
        True
    """
    if base_dir is None:
        data_dir = Path.cwd() / DEFAULT_DATA_DIR
    else:
        data_dir = Path(base_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir.resolve()


def download_rico(
    dataset_type: RicoDatasetType = "semantic",
    data_dir: Path | str | None = None,
    force: bool = False,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> CorpusDataset:
    """Download a Rico dataset to the specified directory.

    This function downloads the specified Rico dataset subset from
    interactionmining.org and extracts it to the data directory.

    Args:
        dataset_type: Which Rico dataset to download. Options:
            - "semantic": Semantic annotations (150MB, recommended)
            - "ui_screenshots": Full screenshots (6GB)
            - "ui_metadata": UI metadata CSV (2MB)
            - "ui_vectors": Layout vectors (8MB)
        data_dir: Target directory for data. Defaults to ./data
        force: If True, re-download even if data exists.
        progress_callback: Optional callback(bytes_downloaded, total_bytes)

    Returns:
        CorpusDataset: Metadata about the downloaded dataset.

    Raises:
        ValueError: If dataset_type is not recognized.
        ConnectionError: If download fails.

    Example:
        >>> dataset = download_rico("semantic", data_dir="./data")
        >>> dataset.downloaded
        True
        >>> dataset.path.exists()
        True
    """
    if dataset_type not in RICO_DATASETS:
        available = ", ".join(RICO_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. Available: {available}"
        )

    dataset_info = RICO_DATASETS[dataset_type]
    url = dataset_info["url"]

    # Determine target directory
    target_dir = get_data_dir(data_dir)
    rico_dir = target_dir / "rico"
    rico_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename from URL
    filename = url.split("/")[-1]
    output_path = rico_dir / filename
    extract_dir = rico_dir / dataset_type

    # Check if already downloaded
    if extract_dir.exists() and not force:
        print(f"Dataset already exists at {extract_dir}")
        return CorpusDataset(
            name=f"rico_{dataset_type}",
            path=extract_dir,
            source="rico",
            dataset_type=dataset_type,
            downloaded=True,
        )

    # Download the file
    print(f"Downloading Rico {dataset_type} dataset ({dataset_info['size_mb']}MB)...")
    print(f"  URL: {url}")
    print(f"  Target: {output_path}")

    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if progress_callback:
            progress_callback(downloaded, total_size)
        elif total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

    try:
        urlretrieve(url, output_path, reporthook=_progress_hook)
        print()  # Newline after progress
    except Exception as e:
        raise ConnectionError(f"Failed to download {url}: {e}") from e

    # Extract if archive
    if output_path.suffix in (".gz", ".zip"):
        print(f"Extracting to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)

        if output_path.name.endswith(".tar.gz"):
            with tarfile.open(output_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        elif output_path.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(output_path, "r") as zf:
                zf.extractall(path=extract_dir)

        # Optionally remove the archive after extraction
        # output_path.unlink()

    print(f"Dataset ready at {extract_dir}")

    return CorpusDataset(
        name=f"rico_{dataset_type}",
        path=extract_dir,
        source="rico",
        dataset_type=dataset_type,
        downloaded=True,
    )


def list_rico_datasets() -> dict[str, dict]:
    """List available Rico datasets and their metadata.

    Returns:
        dict: Mapping of dataset type to metadata (url, size_mb, description).

    Example:
        >>> datasets = list_rico_datasets()
        >>> datasets["semantic"]["size_mb"]
        150
    """
    return RICO_DATASETS.copy()
