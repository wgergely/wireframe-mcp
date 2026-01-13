"""Unit tests for ShowUI provider."""

import json

import pytest

from src.corpus.provider.base.test import BaseProviderTest
from src.corpus.provider.showui import Provider
from src.mid import LayoutNode, is_valid


class TestShowUIProvider(BaseProviderTest):
    """Test suites for ShowUI Provider."""

    @pytest.fixture
    def provider_class(self):
        return Provider

    @pytest.mark.unit
    def test_name_is_showui(self, provider):
        """Verify the provider name is correct."""
        assert provider.name == "showui"

    @pytest.mark.unit
    def test_dest_dir_path(self, provider, tmp_path):
        """Verify destination directory path is correct."""
        assert provider._dest_dir == tmp_path / "showui"

    @pytest.mark.unit
    def test_samples_dir_path(self, provider, tmp_path):
        """Verify samples directory path is correct."""
        assert provider._samples_dir == tmp_path / "showui" / "samples"

    @pytest.mark.unit
    def test_has_data_returns_false_when_empty(self, provider):
        """Verify _has_data returns False when no JSON files exist."""
        provider._samples_dir.mkdir(parents=True, exist_ok=True)
        assert provider.has_data() is False

    @pytest.mark.unit
    def test_has_data_returns_true_when_json_exists(self, provider):
        """Verify _has_data returns True when JSON samples exist."""
        provider._samples_dir.mkdir(parents=True, exist_ok=True)
        (provider._samples_dir / "sample_00000.json").write_text("{}")
        assert provider.has_data() is True

    @pytest.mark.unit
    def test_detection_to_hierarchy(self, provider):
        """Verify flat detections are converted to pseudo-hierarchy."""
        data = {
            "instruction": "Click the submit button",
            "detections": [
                {"label": "action", "bounding_box": [0.1, 0.2, 0.3, 0.4]},
                {"label": "action", "bounding_box": [0.5, 0.6, 0.1, 0.1]},
            ],
        }

        hierarchy = provider._detection_to_hierarchy(data)

        assert hierarchy["type"] == "screen"
        assert hierarchy["id"] == "root"
        assert len(hierarchy["children"]) == 2
        assert hierarchy["children"][0]["type"] == "element"
        assert hierarchy["children"][0]["text"] == "Click the submit button"
        assert hierarchy["children"][0]["bounds"] == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.unit
    def test_process_yields_standardized_data(self, provider):
        """Verify process yields StandardizedData items."""
        from src.corpus.provider.base import StandardizedData

        # Create mock sample
        provider._samples_dir.mkdir(parents=True, exist_ok=True)
        sample_data = {
            "instruction": "Click the menu",
            "detections": [{"label": "action", "bounding_box": [0.1, 0.2, 0.1, 0.1]}],
            "query_type": "appearance",
        }
        sample_path = provider._samples_dir / "sample_00000.json"
        sample_path.write_text(json.dumps(sample_data))

        items = list(provider.process())
        assert len(items) == 1
        assert isinstance(items[0], StandardizedData)
        assert items[0].id == "sample_00000"
        assert items[0].source == "showui"
        assert items[0].dataset == "desktop"

    @pytest.mark.unit
    def test_process_creates_flat_hierarchy(self, provider):
        """Verify process creates flat hierarchy (depth=1)."""
        provider._samples_dir.mkdir(parents=True, exist_ok=True)
        sample_data = {
            "instruction": "Select file",
            "detections": [
                {"label": "action", "bounding_box": [0.1, 0.2, 0.1, 0.1]},
                {"label": "action", "bounding_box": [0.3, 0.4, 0.1, 0.1]},
            ],
        }
        (provider._samples_dir / "sample.json").write_text(json.dumps(sample_data))

        items = list(provider.process())
        hierarchy = items[0].hierarchy

        # Verify flat structure: root with direct children, no nesting
        assert hierarchy["type"] == "screen"
        assert len(hierarchy["children"]) == 2
        for child in hierarchy["children"]:
            assert "children" not in child  # No nested children

    @pytest.mark.unit
    def test_metadata_includes_query_type(self, provider):
        """Verify metadata includes query_type and interfaces."""
        provider._samples_dir.mkdir(parents=True, exist_ok=True)
        sample_data = {
            "instruction": "Click button",
            "detections": [],
            "query_type": "appearance",
            "interfaces": "visual",
        }
        (provider._samples_dir / "test.json").write_text(json.dumps(sample_data))

        items = list(provider.process())
        assert items[0].metadata["query_type"] == "appearance"
        assert items[0].metadata["interfaces"] == "visual"

    @pytest.mark.unit
    def test_process_populates_layout_field(self, provider):
        """Verify process populates the layout field with LayoutNode."""
        provider._samples_dir.mkdir(parents=True, exist_ok=True)
        sample_data = {
            "instruction": "Click the submit button",
            "detections": [
                {"label": "button", "bounding_box": [0.1, 0.2, 0.3, 0.4]},
            ],
        }
        (provider._samples_dir / "test.json").write_text(json.dumps(sample_data))

        items = list(provider.process())
        assert items[0].layout is not None
        assert isinstance(items[0].layout, LayoutNode)
        assert is_valid(items[0].layout)

    @pytest.mark.unit
    def test_layout_has_correct_component_types(self, provider):
        """Verify layout infers correct component types from labels."""
        provider._samples_dir.mkdir(parents=True, exist_ok=True)
        sample_data = {
            "instruction": "Test various element types",
            "detections": [
                {"label": "button", "bounding_box": [0.1, 0.1, 0.1, 0.05]},
                {"label": "input", "bounding_box": [0.1, 0.2, 0.3, 0.05]},
                {"label": "icon", "bounding_box": [0.1, 0.3, 0.05, 0.05]},
            ],
        }
        (provider._samples_dir / "types.json").write_text(json.dumps(sample_data))

        items = list(provider.process())
        layout = items[0].layout

        # Check component types (type field stores string value due to Pydantic)
        assert layout.children[0].type == "button"
        assert layout.children[1].type == "input"
        assert layout.children[2].type == "icon"


class TestShowUIHierarchyToLayout:
    """Tests for ShowUI hierarchy to layout conversion."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create a provider instance for testing."""
        return Provider(tmp_path)

    @pytest.mark.unit
    def test_converts_normalized_width_to_flex_ratio(self, provider):
        """Verify normalized width [x, y, w, h] is converted to flex ratio."""
        hierarchy = {
            "type": "screen",
            "id": "root",
            "children": [
                {
                    "type": "element",
                    "id": "element_0",
                    "bounds": [0.1, 0.2, 0.5, 0.4],  # width=0.5 → flex_ratio=6
                    "label": "action",
                }
            ],
        }

        layout = provider._showui_hierarchy_to_layout(hierarchy)

        child = layout.children[0]
        # width_ratio=0.5 → flex_ratio = round(0.5 * 12) = 6
        assert child.flex_ratio == 6

    @pytest.mark.unit
    def test_root_has_full_flex_ratio(self, provider):
        """Verify root node has full flex ratio (12)."""
        hierarchy = {"type": "screen", "children": []}

        layout = provider._showui_hierarchy_to_layout(hierarchy)

        assert layout.flex_ratio == 12

    @pytest.mark.unit
    def test_infers_button_from_label(self, provider):
        """Verify button label maps to BUTTON component type."""
        hierarchy = {
            "children": [
                {"bounds": [0.1, 0.1, 0.1, 0.1], "label": "button"},
                {"bounds": [0.2, 0.2, 0.1, 0.1], "label": "click_here"},
            ]
        }

        layout = provider._showui_hierarchy_to_layout(hierarchy)

        # LayoutNode stores string values due to use_enum_values=True
        assert layout.children[0].type == "button"
        assert layout.children[1].type == "button"

    @pytest.mark.unit
    def test_empty_hierarchy(self, provider):
        """Verify empty hierarchy produces valid layout."""
        hierarchy = {"type": "screen", "children": []}

        layout = provider._showui_hierarchy_to_layout(hierarchy)

        assert is_valid(layout)
        assert len(layout.children) == 0

    @pytest.mark.unit
    def test_preserves_text_from_hierarchy(self, provider):
        """Verify text is preserved in layout nodes as label."""
        hierarchy = {
            "type": "screen",
            "text": "Main instruction",
            "children": [
                {
                    "bounds": [0.1, 0.1, 0.1, 0.1],
                    "text": "Element text",
                    "label": "action",
                }
            ],
        }

        layout = provider._showui_hierarchy_to_layout(hierarchy)

        assert layout.label == "Main instruction"
        assert layout.children[0].label == "Element text"
