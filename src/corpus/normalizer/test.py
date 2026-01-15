"""Unit tests for the normalizer module."""

import pytest

from src.corpus.normalizer import (
    ANDROID_CLASS_MAP,
    COMPONENT_LABEL_MAP,
    count_components,
    extract_text_content,
    hierarchy_to_layout,
    node_count,
    normalize_enrico_hierarchy,
    normalize_rico_hierarchy,
    tree_depth,
)
from src.mid import ComponentType, LayoutNode, Orientation, is_valid


@pytest.fixture
def simple_hierarchy() -> dict:
    """Simple hierarchy with componentLabel."""
    return {
        "class": "android.widget.FrameLayout",
        "bounds": [0, 0, 1080, 1920],
        "componentLabel": "Card",
        "children": [
            {
                "class": "android.widget.TextView",
                "bounds": [10, 10, 500, 100],
                "text": "Hello World",
                "componentLabel": "Text",
                "children": [],
            },
            {
                "class": "android.widget.Button",
                "bounds": [10, 200, 500, 300],
                "text": "Click Me",
                "componentLabel": "Text Button",
                "children": [],
            },
        ],
    }


@pytest.fixture
def nested_hierarchy() -> dict:
    """Deeply nested hierarchy for depth testing."""
    return {
        "class": "android.view.ViewGroup",
        "bounds": [0, 0, 1080, 1920],
        "children": [
            {
                "class": "android.widget.LinearLayout",
                "bounds": [0, 0, 1080, 500],
                "children": [
                    {
                        "class": "android.widget.FrameLayout",
                        "bounds": [0, 0, 540, 250],
                        "children": [
                            {
                                "class": "android.widget.TextView",
                                "bounds": [10, 10, 530, 240],
                                "text": "Deep text",
                                "componentLabel": "Text",
                            }
                        ],
                    }
                ],
            }
        ],
    }


@pytest.fixture
def horizontal_layout_hierarchy() -> dict:
    """Hierarchy with horizontally arranged children."""
    return {
        "class": "android.widget.LinearLayout",
        "bounds": [0, 0, 1080, 100],
        "children": [
            {"class": "android.widget.Button", "bounds": [0, 0, 360, 100]},
            {"class": "android.widget.Button", "bounds": [360, 0, 720, 100]},
            {"class": "android.widget.Button", "bounds": [720, 0, 1080, 100]},
        ],
    }


class TestComponentLabelMapping:
    """Tests for componentLabel to ComponentType mapping."""

    @pytest.mark.unit
    def test_all_rico_labels_mapped(self):
        """All expected Rico labels should be in the mapping."""
        rico_labels = [
            "Advertisement",
            "Background Image",
            "Card",
            "Drawer",
            "Icon",
            "Image",
            "Input",
            "List Item",
            "Map View",
            "Modal",
            "Multi-Tab",
            "On/Off Switch",
            "Radio Button",
            "Text",
            "Text Button",
            "Toolbar",
            "Web View",
        ]
        for label in rico_labels:
            assert label in COMPONENT_LABEL_MAP, f"Missing mapping for {label}"

    @pytest.mark.unit
    def test_label_maps_to_valid_component_type(self):
        """All mappings should produce valid ComponentType values."""
        for _label, comp_type in COMPONENT_LABEL_MAP.items():
            assert isinstance(comp_type, ComponentType)


class TestAndroidClassMapping:
    """Tests for Android class fallback mapping."""

    @pytest.mark.unit
    def test_common_widgets_mapped(self):
        """Common Android widgets should be in the mapping."""
        common_widgets = ["Button", "TextView", "ImageView", "EditText", "CheckBox"]
        for widget in common_widgets:
            assert widget in ANDROID_CLASS_MAP, f"Missing mapping for {widget}"

    @pytest.mark.unit
    def test_class_maps_to_valid_component_type(self):
        """All mappings should produce valid ComponentType values."""
        for _class_name, comp_type in ANDROID_CLASS_MAP.items():
            assert isinstance(comp_type, ComponentType)


class TestHierarchyToLayout:
    """Tests for hierarchy_to_layout conversion."""

    @pytest.mark.unit
    def test_simple_conversion(self, simple_hierarchy):
        """Simple hierarchy converts to valid LayoutNode."""
        layout = hierarchy_to_layout(simple_hierarchy)

        assert isinstance(layout, LayoutNode)
        assert layout.type == ComponentType.CARD.value
        assert len(layout.children) == 2

    @pytest.mark.unit
    def test_produces_valid_layout(self, simple_hierarchy):
        """Converted layout passes validation."""
        layout = hierarchy_to_layout(simple_hierarchy)
        assert is_valid(layout)

    @pytest.mark.unit
    def test_preserves_text_labels(self, simple_hierarchy):
        """Text content is preserved as labels."""
        layout = hierarchy_to_layout(simple_hierarchy)

        child_labels = [c.label for c in layout.children]
        assert "Hello World" in child_labels
        assert "Click Me" in child_labels

    @pytest.mark.unit
    def test_generates_unique_ids(self, simple_hierarchy):
        """Each node gets a unique ID."""
        layout = hierarchy_to_layout(simple_hierarchy)

        all_ids = set()

        def collect_ids(node):
            all_ids.add(node.id)
            for child in node.children:
                collect_ids(child)

        collect_ids(layout)

        # Should have 3 unique IDs (root + 2 children)
        assert len(all_ids) == 3

    @pytest.mark.unit
    def test_infers_horizontal_orientation(self, horizontal_layout_hierarchy):
        """Horizontal child arrangement is detected."""
        layout = hierarchy_to_layout(horizontal_layout_hierarchy)
        assert layout.orientation == Orientation.HORIZONTAL.value

    @pytest.mark.unit
    def test_default_vertical_orientation(self, simple_hierarchy):
        """Default orientation is vertical."""
        layout = hierarchy_to_layout(simple_hierarchy)
        assert layout.orientation == Orientation.VERTICAL.value


class TestFlexRatioCalculation:
    """Tests for flex ratio calculation."""

    @pytest.mark.unit
    def test_full_width_is_12(self):
        """Full-width child gets flex ratio 12."""
        hierarchy = {
            "bounds": [0, 0, 1080, 1920],
            "children": [{"bounds": [0, 0, 1080, 100]}],
        }
        layout = hierarchy_to_layout(hierarchy)
        assert layout.children[0].flex_ratio == 12

    @pytest.mark.unit
    def test_half_width_is_6(self):
        """Half-width child gets flex ratio 6."""
        hierarchy = {
            "bounds": [0, 0, 1080, 1920],
            "children": [{"bounds": [0, 0, 540, 100]}],
        }
        layout = hierarchy_to_layout(hierarchy)
        assert layout.children[0].flex_ratio == 6

    @pytest.mark.unit
    def test_flex_ratio_clamped_to_range(self):
        """Flex ratio is clamped to 1-12."""
        # Tiny element
        hierarchy = {
            "bounds": [0, 0, 1080, 1920],
            "children": [{"bounds": [0, 0, 10, 100]}],
        }
        layout = hierarchy_to_layout(hierarchy)
        assert 1 <= layout.children[0].flex_ratio <= 12


class TestNormalizeRicoHierarchy:
    """Tests for Rico-specific normalization."""

    @pytest.mark.unit
    def test_uses_rico_prefix(self, simple_hierarchy):
        """Rico normalization uses rico_ ID prefix."""
        layout = normalize_rico_hierarchy(simple_hierarchy, "12345")
        assert layout.id.startswith("rico_12345")

    @pytest.mark.unit
    def test_produces_valid_layout(self, simple_hierarchy):
        """Rico normalized layout passes validation."""
        layout = normalize_rico_hierarchy(simple_hierarchy, "test")
        assert is_valid(layout)


class TestNormalizeEnricoHierarchy:
    """Tests for Enrico-specific normalization."""

    @pytest.mark.unit
    def test_uses_enrico_prefix(self, simple_hierarchy):
        """Enrico normalization uses enrico_ ID prefix."""
        layout = normalize_enrico_hierarchy(simple_hierarchy, "12345")
        assert layout.id.startswith("enrico_12345")

    @pytest.mark.unit
    def test_produces_valid_layout(self, simple_hierarchy):
        """Enrico normalized layout passes validation."""
        layout = normalize_enrico_hierarchy(simple_hierarchy, "test")
        assert is_valid(layout)


class TestTreeAnalysisFunctions:
    """Tests for tree analysis utilities."""

    @pytest.mark.unit
    def test_count_components(self, simple_hierarchy):
        """count_components returns correct counts."""
        layout = hierarchy_to_layout(simple_hierarchy)
        counts = count_components(layout)

        assert counts.get("card") == 1
        assert counts.get("text") == 1
        assert counts.get("text_button") == 1

    @pytest.mark.unit
    def test_extract_text_content(self, simple_hierarchy):
        """extract_text_content returns all text labels."""
        layout = hierarchy_to_layout(simple_hierarchy)
        texts = extract_text_content(layout)

        assert "Hello World" in texts
        assert "Click Me" in texts

    @pytest.mark.unit
    def test_tree_depth(self, nested_hierarchy):
        """tree_depth returns correct depth."""
        layout = hierarchy_to_layout(nested_hierarchy)
        depth = tree_depth(layout)

        # Root -> LinearLayout -> FrameLayout -> TextView = 4
        assert depth == 4

    @pytest.mark.unit
    def test_node_count(self, simple_hierarchy):
        """node_count returns total nodes."""
        layout = hierarchy_to_layout(simple_hierarchy)
        count = node_count(layout)

        # 1 root + 2 children = 3
        assert count == 3


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_hierarchy(self):
        """Empty hierarchy produces valid container."""
        hierarchy = {}
        layout = hierarchy_to_layout(hierarchy)

        assert isinstance(layout, LayoutNode)
        assert layout.type == ComponentType.CONTAINER.value
        assert layout.children == []

    @pytest.mark.unit
    def test_no_children(self):
        """Hierarchy with no children works."""
        hierarchy = {
            "class": "android.widget.TextView",
            "text": "Solo text",
            "componentLabel": "Text",
        }
        layout = hierarchy_to_layout(hierarchy)

        assert layout.type == ComponentType.TEXT.value
        assert layout.label == "Solo text"
        assert layout.children == []

    @pytest.mark.unit
    def test_missing_bounds(self):
        """Missing bounds doesn't crash."""
        hierarchy = {
            "class": "android.widget.FrameLayout",
            "children": [{"class": "android.widget.TextView"}],
        }
        layout = hierarchy_to_layout(hierarchy)

        assert isinstance(layout, LayoutNode)
        assert layout.flex_ratio == 1  # Default

    @pytest.mark.unit
    def test_unknown_component_label(self):
        """Unknown componentLabel falls back to class-based inference."""
        hierarchy = {
            "class": "android.widget.Button",
            "componentLabel": "UnknownWidget",
        }
        layout = hierarchy_to_layout(hierarchy)

        # Should fall back to Button class mapping
        assert layout.type == ComponentType.BUTTON.value
