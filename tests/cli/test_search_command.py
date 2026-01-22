"""Tests for search command environment handling."""

import subprocess
import sys


def test_search_accepts_docker_flag():
    """search command should accept --docker flag."""
    result = subprocess.run(
        [sys.executable, ".", "search", "--help"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
    )
    assert "--docker" in result.stdout


def test_search_accepts_force_flag():
    """search command should accept --force flag."""
    result = subprocess.run(
        [sys.executable, ".", "search", "--help"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
    )
    assert "--force" in result.stdout


def test_search_accepts_image_flag():
    """search command should accept --image flag for Docker image."""
    result = subprocess.run(
        [sys.executable, ".", "search", "--help"],
        capture_output=True,
        text=True,
        cwd="Y:\\code\\wireframe-mcp-worktrees\\feature-cli-env-audit",
    )
    assert "--image" in result.stdout
