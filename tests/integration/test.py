"""Integration tests for GPU-accelerated indexing."""

import subprocess
from pathlib import Path

import pytest


def _has_nvidia_docker() -> bool:
    """Check if NVIDIA Docker runtime is available."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
        )
        return "nvidia" in result.stdout
    except Exception:
        return False


def _has_wireframe_mcp_image() -> bool:
    """Check if wireframe-mcp Docker image exists and is functional.

    Validates that:
    1. The image exists
    2. Python can be executed inside the container
    """
    try:
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", "wireframe-mcp:latest"],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            return False

        # Verify the image can run Python (validates it's properly configured)
        # Note: The image has ENTRYPOINT ["python"], so we pass args to python directly
        result = subprocess.run(
            ["docker", "run", "--rm", "wireframe-mcp:latest", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


# Get the project root directory (three levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.docker
@pytest.mark.skipif(
    not _has_nvidia_docker(),
    reason="NVIDIA Docker runtime not available",
)
@pytest.mark.skipif(
    not _has_wireframe_mcp_image(),
    reason="wireframe-mcp:latest Docker image not available",
)
class TestGpuIndexBuild:
    """Integration tests for GPU-accelerated Docker index building."""

    def test_docker_index_build(self):
        """Test full Docker-based index build with GPU support.

        This test verifies that:
        1. The Docker-based index build command runs successfully
        2. GPU/FAISS-GPU is being used (detected via output)
        """
        result = subprocess.run(
            [
                "python",
                ".",
                "index",
                "build",
                "rico_semantic",
                "--docker",
                "--limit",
                "50",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        # Verify GPU is mentioned in output (indicates FAISS-GPU usage)
        output_lower = (result.stdout + result.stderr).lower()
        assert "gpu" in output_lower, f"GPU not mentioned in output: {result.stdout}"
