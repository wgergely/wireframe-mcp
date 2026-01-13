"""Standardized download utilities with progress reporting.

Provides consistent download experience across all corpus providers with:
- Progress bar visualization (tqdm)
- File size estimation
- Standardized INFO-level logging
- Archive extraction support
"""

import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from src.core import get_logger

logger = get_logger("corpus.download")

# Default chunk size for streaming downloads (8KB)
CHUNK_SIZE = 8192


def download_file(
    url: str,
    dest_path: Path,
    provider_name: str,
    expected_size_mb: int | None = None,
) -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download from.
        dest_path: Local path to save the file.
        provider_name: Name of the provider (for logging).
        expected_size_mb: Expected file size in MB (for display if server
            doesn't provide Content-Length).

    Raises:
        ConnectionError: If download fails.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed. Install with: pip install tqdm")
        _download_file_simple(url, dest_path, provider_name)
        return

    logger.info(f"[{provider_name}] Downloading from {url}")

    try:
        request = Request(url, headers={"User-Agent": "wireframe-mcp/1.0"})
        response = urlopen(request)

        # Get file size from headers or use estimate
        total_size = response.headers.get("Content-Length")
        if total_size:
            total_size = int(total_size)
        elif expected_size_mb:
            total_size = expected_size_mb * 1024 * 1024
            logger.info(f"[{provider_name}] Estimated size: ~{expected_size_mb}MB")
        else:
            total_size = None

        # Create progress bar
        desc = f"[{provider_name}]"
        with (
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                ncols=80,
            ) as pbar,
            open(dest_path, "wb") as f,
        ):
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info(f"[{provider_name}] Downloaded to {dest_path}")

    except Exception as e:
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        raise ConnectionError(f"[{provider_name}] Download failed: {e}") from e


def _download_file_simple(url: str, dest_path: Path, provider_name: str) -> None:
    """Simple download fallback without progress bar.

    Args:
        url: URL to download from.
        dest_path: Local path to save the file.
        provider_name: Name of the provider (for logging).
    """
    from urllib.request import urlretrieve

    logger.info(f"[{provider_name}] Downloading (no progress bar)...")
    urlretrieve(url, dest_path)
    logger.info(f"[{provider_name}] Downloaded to {dest_path}")


def extract_archive(
    archive_path: Path,
    extract_dir: Path,
    provider_name: str,
) -> None:
    """Extract an archive with progress logging.

    Supports .tar.gz and .zip formats.

    Args:
        archive_path: Path to the archive file.
        extract_dir: Directory to extract contents into.
        provider_name: Name of the provider (for logging).

    Raises:
        ValueError: If archive format is not supported.
    """
    logger.info(f"[{provider_name}] Extracting {archive_path.name}...")

    extract_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=extract_dir)
    elif archive_path.suffix == ".csv":
        # CSV files don't need extraction, just copy
        import shutil

        shutil.copy(archive_path, extract_dir / archive_path.name)
    else:
        raise ValueError(
            f"[{provider_name}] Unsupported archive format: {archive_path.suffix}"
        )

    logger.info(f"[{provider_name}] Extracted to {extract_dir}")


def download_and_extract(
    url: str,
    dest_dir: Path,
    extract_dir: Path,
    provider_name: str,
    expected_size_mb: int | None = None,
) -> None:
    """Download and extract an archive in one step.

    Args:
        url: URL to download from.
        dest_dir: Directory to save the downloaded archive.
        extract_dir: Directory to extract contents into.
        provider_name: Name of the provider (for logging).
        expected_size_mb: Expected file size in MB.
    """
    filename = url.split("/")[-1]
    archive_path = dest_dir / filename

    download_file(url, archive_path, provider_name, expected_size_mb)
    extract_archive(archive_path, extract_dir, provider_name)


__all__ = ["download_file", "extract_archive", "download_and_extract"]
