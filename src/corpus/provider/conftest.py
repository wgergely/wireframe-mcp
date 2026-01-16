"""Corpus provider test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def mock_corpus_dir(tmp_path: Path) -> Path:
    """Create mock corpus directory structure for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the mock corpus data directory.
    """
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()

    # Create mock JSON files simulating RICO-like data
    (data_dir / "screen1.json").write_text(
        """{
        "activity_name": "MainActivity",
        "children": [
            {"componentLabel": "button", "bounds": [0, 0, 100, 50]},
            {"componentLabel": "text", "bounds": [0, 50, 200, 100]}
        ]
    }"""
    )

    (data_dir / "screen2.json").write_text(
        """{
        "activity_name": "SettingsActivity",
        "children": [
            {"componentLabel": "list", "bounds": [0, 0, 400, 800]}
        ]
    }"""
    )

    return data_dir


@pytest.fixture
def mock_screenshot_dir(tmp_path: Path) -> Path:
    """Create mock screenshot directory with placeholder images.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the mock screenshot directory.
    """
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()

    # Create minimal PNG placeholders (1x1 pixel)
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    (screenshot_dir / "screen1.png").write_bytes(minimal_png)
    (screenshot_dir / "screen2.png").write_bytes(minimal_png)

    return screenshot_dir


@pytest.fixture
def sample_hierarchy() -> dict:
    """Sample Android hierarchy structure for normalization tests.

    Returns:
        A dictionary representing an Android UI hierarchy.
    """
    return {
        "activity_name": "TestActivity",
        "bounds": [0, 0, 1080, 1920],
        "componentLabel": "container",
        "children": [
            {
                "bounds": [0, 0, 1080, 100],
                "componentLabel": "toolbar",
                "children": [
                    {"bounds": [20, 20, 60, 60], "componentLabel": "button"},
                    {"bounds": [80, 20, 500, 60], "componentLabel": "text"},
                ],
            },
            {
                "bounds": [0, 100, 1080, 1800],
                "componentLabel": "list",
                "children": [
                    {"bounds": [0, 100, 1080, 300], "componentLabel": "list_item"},
                    {"bounds": [0, 300, 1080, 500], "componentLabel": "list_item"},
                ],
            },
        ],
    }


@pytest.fixture
def nested_hierarchy() -> dict:
    """Deeply nested hierarchy for depth testing.

    Returns:
        A deeply nested hierarchy structure.
    """
    # Create 5-level deep hierarchy
    hierarchy = {"componentLabel": "root", "bounds": [0, 0, 1080, 1920], "children": []}

    current = hierarchy
    for i in range(5):
        child = {
            "componentLabel": f"level_{i}",
            "bounds": [i * 10, i * 10, 1000 - i * 10, 1800 - i * 10],
            "children": [],
        }
        current["children"].append(child)
        current = child

    return hierarchy


@pytest.fixture
def flat_hierarchy() -> dict:
    """Flat hierarchy (depth=1) for edge case testing.

    Returns:
        A flat hierarchy with no nesting.
    """
    return {
        "componentLabel": "container",
        "bounds": [0, 0, 1080, 1920],
        "children": [
            {"componentLabel": "button", "bounds": [0, 0, 100, 50]},
            {"componentLabel": "text", "bounds": [110, 0, 400, 50]},
            {"componentLabel": "image", "bounds": [0, 60, 200, 200]},
        ],
    }
