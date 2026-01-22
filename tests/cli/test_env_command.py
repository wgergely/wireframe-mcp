"""Tests for env CLI command."""

import subprocess
import sys


def test_env_check_runs_without_error():
    """env check command should run and exit cleanly."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    # Should exit 0 (has GPU) or 1 (no GPU), but not crash
    assert result.returncode in (0, 1)
    assert "Environment Report" in result.stdout or "Platform:" in result.stdout


def test_env_check_shows_platform():
    """env check should show platform information."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert "Platform:" in result.stdout


def test_env_check_shows_gpu():
    """env check should show GPU information."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert "GPU:" in result.stdout


def test_env_check_shows_docker():
    """env check should show Docker information."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert "Docker:" in result.stdout


def test_env_check_accepts_docker_flag():
    """env check should accept --docker flag."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check", "--help"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert "--docker" in result.stdout


def test_env_check_accepts_image_flag():
    """env check should accept --image flag."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check", "--help"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert "--image" in result.stdout


def test_env_gpu_subcommand():
    """env gpu subcommand should work."""
    result = subprocess.run(
        [sys.executable, ".", "env", "gpu"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert result.returncode in (0, 1)
    assert "GPU:" in result.stdout or "FAISS" in result.stdout


def test_env_gpu_accepts_docker_flag():
    """env gpu should accept --docker flag."""
    result = subprocess.run(
        [sys.executable, ".", "env", "gpu", "--help"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert "--docker" in result.stdout


def test_env_docker_subcommand():
    """env docker subcommand should work."""
    result = subprocess.run(
        [sys.executable, ".", "env", "docker"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
        timeout=30,
    )
    assert result.returncode in (0, 1)
    assert "Docker:" in result.stdout
