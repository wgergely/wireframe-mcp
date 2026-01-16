"""Render module test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_d2_dsl() -> str:
    """Sample D2 DSL for testing.

    Returns:
        A valid D2 diagram source string.
    """
    return """
root: Dashboard {
    direction: right
    sidebar: Navigation { }
    main: Content {
        header: Title { }
        body: Data { }
    }
}
"""


@pytest.fixture
def sample_plantuml_dsl() -> str:
    """Sample PlantUML Salt DSL for testing.

    Returns:
        A valid PlantUML Salt diagram source string.
    """
    return """@startsalt
{
    "Dashboard"
    { [Button 1] | [Button 2] }
    ---
    {
        "Input: " | "___________"
    }
}
@endsalt"""


@pytest.fixture
def simple_plantuml_dsl() -> str:
    """Minimal PlantUML DSL for quick render tests.

    Returns:
        A minimal valid PlantUML Salt diagram.
    """
    return """@startsalt
{
    [OK]
}
@endsalt"""
