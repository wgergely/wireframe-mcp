"""Core reset functionality for wireframe-mcp development environment.

This module provides functions to clean up development artifacts,
supporting testing workflows that require a clean slate.

Functions:
    clear_indices: Remove RAG vector indices
    clear_models: Remove downloaded embedding models
    clear_docker: Purge Docker containers, volumes, and images
    clear_temp: Remove temporary files and caches
    reset_environment: Orchestrate full or partial reset
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.config import find_repo_root, get_index_dir, get_models_dir
from docker.lib import (
    PROJECT_PREFIX,
    Service,
    get_service_info,
    VOLUME_CORPUS_DATA,
    VOLUME_CORPUS_MODELS,
    VOLUME_OUTPUT,
)

__all__ = [
    "clear_indices",
    "clear_models",
    "clear_docker",
    "clear_temp",
    "reset_environment",
]


def _print(msg: str, verbose: bool = True) -> None:
    """Print message if verbose mode enabled."""
    if verbose:
        print(msg)


def _delete_path(path: Path, verbose: bool = True) -> bool:
    """Delete a file or directory.

    Args:
        path: Path to delete.
        verbose: Print status messages.

    Returns:
        True if deleted or didn't exist, False on error.
    """
    if not path.exists():
        return True

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        _print(f"  Deleted {path}", verbose)
        return True
    except OSError as e:
        _print(f"  Failed to delete {path}: {e}", verbose)
        return False


def clear_indices(verbose: bool = True) -> bool:
    """Delete RAG vector indices.

    Removes:
        - .corpus/index/ contents (FAISS index, metadata)
        - data/test-index/ (pytest fixtures)

    Args:
        verbose: Print progress messages.

    Returns:
        True if all deletions succeeded.
    """
    _print("[reset] Clearing RAG indices...", verbose)
    success = True

    # Main index directory
    try:
        index_dir = get_index_dir()
        if index_dir.exists():
            for item in index_dir.iterdir():
                if not _delete_path(item, verbose):
                    success = False
        else:
            _print(f"  Index directory does not exist: {index_dir}", verbose)
    except Exception as e:
        _print(f"  Failed to access index directory: {e}", verbose)
        success = False

    # Test index directory
    try:
        repo_root = find_repo_root()
        test_index = repo_root / "data" / "test-index"
        if test_index.exists():
            if not _delete_path(test_index, verbose):
                success = False
        else:
            _print("  Test index directory does not exist", verbose)
    except Exception as e:
        _print(f"  Failed to access test index: {e}", verbose)
        success = False

    return success


def clear_models(verbose: bool = True) -> bool:
    """Delete downloaded embedding models.

    Removes:
        - .corpus/models/ contents (sentence-transformers, etc.)

    Models are re-downloaded on next index build.

    Args:
        verbose: Print progress messages.

    Returns:
        True if all deletions succeeded.
    """
    _print("[reset] Clearing embedding models...", verbose)
    success = True

    try:
        models_dir = get_models_dir()
        if models_dir.exists():
            for item in models_dir.iterdir():
                if not _delete_path(item, verbose):
                    success = False
        else:
            _print(f"  Models directory does not exist: {models_dir}", verbose)
    except Exception as e:
        _print(f"  Failed to access models directory: {e}", verbose)
        success = False

    return success


def _run_docker_command(args: list[str], verbose: bool = True) -> tuple[bool, str]:
    """Run a docker command and return success status.

    Args:
        args: Command arguments (without 'docker' prefix).
        verbose: Print command output on error.

    Returns:
        Tuple of (success, output).
    """
    cmd = ["docker"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "Docker not found"
    except Exception as e:
        return False, str(e)


def _get_container_ids(prefix: str) -> list[str]:
    """Get container IDs matching prefix.

    Args:
        prefix: Container name prefix to match.

    Returns:
        List of container IDs.
    """
    success, output = _run_docker_command(
        ["ps", "-a", "--filter", f"name={prefix}", "--format", "{{.ID}}"],
        verbose=False,
    )
    if not success or not output.strip():
        return []
    return output.strip().split("\n")


def _get_image_ids(prefix: str) -> list[tuple[str, str]]:
    """Get image IDs matching prefix.

    Args:
        prefix: Image name prefix to match.

    Returns:
        List of (image_id, image_name) tuples.
    """
    success, output = _run_docker_command(
        ["images", "--format", "{{.ID}}\t{{.Repository}}:{{.Tag}}"],
        verbose=False,
    )
    if not success or not output.strip():
        return []

    images = []
    for line in output.strip().split("\n"):
        if "\t" in line:
            img_id, name = line.split("\t", 1)
            if prefix in name:
                images.append((img_id, name))
    return images


def clear_docker(verbose: bool = True) -> bool:
    """Purge Docker containers, volumes, and images.

    Removes:
        - All wfmcp-* containers (stop and remove)
        - Named volumes (wfmcp-corpus-data, wfmcp-corpus-models, wfmcp-output)
        - Project images (wireframe-mcp-*)

    Args:
        verbose: Print progress messages.

    Returns:
        True if all operations succeeded (or Docker unavailable).
    """
    _print("[reset] Purging Docker stack...", verbose)

    # Check if Docker is available
    success, _ = _run_docker_command(["info"], verbose=False)
    if not success:
        _print("  Docker not available, skipping", verbose)
        return True

    all_success = True

    # Stop and remove containers
    container_ids = _get_container_ids(PROJECT_PREFIX)
    if container_ids:
        for cid in container_ids:
            # Stop container
            success, _ = _run_docker_command(["stop", cid], verbose=False)
            if success:
                _print(f"  Stopped container {cid[:12]}", verbose)
            # Remove container
            success, err = _run_docker_command(["rm", "-f", cid], verbose=False)
            if success:
                _print(f"  Removed container {cid[:12]}", verbose)
            else:
                _print(f"  Failed to remove container {cid[:12]}: {err}", verbose)
                all_success = False
    else:
        _print("  No containers to remove", verbose)

    # Remove volumes
    volumes = [VOLUME_CORPUS_DATA, VOLUME_CORPUS_MODELS, VOLUME_OUTPUT]
    for vol in volumes:
        success, err = _run_docker_command(["volume", "rm", "-f", vol], verbose=False)
        if success:
            _print(f"  Removed volume {vol}", verbose)
        elif "No such volume" in err or "not found" in err.lower():
            _print(f"  Volume {vol} does not exist", verbose)
        else:
            _print(f"  Failed to remove volume {vol}: {err}", verbose)
            all_success = False

    # Remove images
    # Get all project-related images
    image_ids = _get_image_ids("wireframe-mcp")
    # Also get kroki images used by the project
    for service in Service:
        info = get_service_info(service)
        if "kroki" in info.image:
            kroki_images = _get_image_ids(info.image.split(":")[0])
            image_ids.extend(kroki_images)

    # Deduplicate
    seen = set()
    unique_images = []
    for img_id, name in image_ids:
        if img_id not in seen:
            seen.add(img_id)
            unique_images.append((img_id, name))

    if unique_images:
        for img_id, name in unique_images:
            success, err = _run_docker_command(["rmi", "-f", img_id], verbose=False)
            if success:
                _print(f"  Removed image {name}", verbose)
            else:
                _print(f"  Failed to remove image {name}: {err}", verbose)
                all_success = False
    else:
        _print("  No images to remove", verbose)

    return all_success


def clear_temp(verbose: bool = True) -> bool:
    """Delete temporary files and caches.

    Removes:
        - .pytest_cache/
        - **/__pycache__/

    Args:
        verbose: Print progress messages.

    Returns:
        True if all deletions succeeded.
    """
    _print("[reset] Clearing temporary files...", verbose)
    success = True

    try:
        repo_root = find_repo_root()
    except Exception as e:
        _print(f"  Failed to find repo root: {e}", verbose)
        return False

    # Pytest cache
    pytest_cache = repo_root / ".pytest_cache"
    if pytest_cache.exists():
        if not _delete_path(pytest_cache, verbose):
            success = False

    # Python caches
    pycache_count = 0
    for pycache in repo_root.rglob("__pycache__"):
        if _delete_path(pycache, verbose=False):
            pycache_count += 1
        else:
            success = False
    if pycache_count > 0:
        _print(f"  Deleted {pycache_count} __pycache__ directories", verbose)

    return success


def reset_environment(
    index: bool = False,
    models: bool = False,
    docker: bool = False,
    temp: bool = False,
    verbose: bool = True,
) -> bool:
    """Reset development environment artifacts.

    Orchestrates cleanup based on flags. If no flags are set,
    defaults to index-only reset.

    Args:
        index: Clear RAG indices.
        models: Clear embedding models.
        docker: Purge Docker stack.
        temp: Clear temporary files.
        verbose: Print progress messages.

    Returns:
        True if all requested operations succeeded.
    """
    # Default to index if nothing specified
    if not any([index, models, docker, temp]):
        index = True

    success = True

    if index:
        if not clear_indices(verbose):
            success = False

    if models:
        if not clear_models(verbose):
            success = False

    if docker:
        if not clear_docker(verbose):
            success = False

    if temp:
        if not clear_temp(verbose):
            success = False

    if success:
        _print("\n[reset] Done. Environment reset complete.", verbose)
    else:
        _print("\n[reset] Completed with errors.", verbose)

    return success
