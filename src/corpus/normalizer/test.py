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
from src.corpus.normalizer.html import (
    HTML_TAG_MAP,
    html_hierarchy_to_layout_hierarchy,
    normalize_html_to_hierarchy,
    parse_html_to_hierarchy,
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


# =============================================================================
# HTML Normalizer Tests
# =============================================================================


class TestParseHtmlToHierarchy:
    """Tests for HTML parsing."""

    @pytest.mark.unit
    def test_simple_html(self):
        """Simple HTML parses correctly."""
        html = "<div><p>Hello</p></div>"
        hierarchy = parse_html_to_hierarchy(html)

        assert hierarchy["tag"] == "div"
        assert len(hierarchy["children"]) == 1
        assert hierarchy["children"][0]["tag"] == "p"

    @pytest.mark.unit
    def test_extracts_body(self):
        """Parser extracts body content."""
        html = (
            "<html><head><title>Test</title></head>"
            "<body><div>Content</div></body></html>"
        )
        hierarchy = parse_html_to_hierarchy(html)

        assert hierarchy["tag"] == "body"
        assert len(hierarchy["children"]) == 1

    @pytest.mark.unit
    def test_preserves_text(self):
        """Text content is preserved."""
        html = "<p>Hello World</p>"
        hierarchy = parse_html_to_hierarchy(html)

        assert hierarchy["text"] == "Hello World"

    @pytest.mark.unit
    def test_extracts_attributes(self):
        """Important attributes are extracted."""
        html = (
            '<input type="text" placeholder="Enter name" '
            'id="name-input" class="form-control">'
        )
        hierarchy = parse_html_to_hierarchy(html)

        assert hierarchy["input_type"] == "text"
        assert hierarchy["placeholder"] == "Enter name"
        assert hierarchy["html_id"] == "name-input"
        assert hierarchy["html_class"] == "form-control"

    @pytest.mark.unit
    def test_skips_script_tags(self):
        """Script and style tags are skipped."""
        html = "<div><script>alert('hi')</script><p>Text</p><style>.x{}</style></div>"
        hierarchy = parse_html_to_hierarchy(html)

        # Should only have the p tag
        assert len(hierarchy["children"]) == 1
        assert hierarchy["children"][0]["tag"] == "p"

    @pytest.mark.unit
    def test_handles_void_tags(self):
        """Void tags (img, input, br) are handled correctly."""
        html = "<div><img src='x.jpg' alt='Image'><input type='text'></div>"
        hierarchy = parse_html_to_hierarchy(html)

        assert len(hierarchy["children"]) == 2
        assert hierarchy["children"][0]["tag"] == "img"
        assert hierarchy["children"][0]["alt"] == "Image"


class TestHtmlHierarchyToLayoutHierarchy:
    """Tests for HTML to Rico-compatible hierarchy conversion."""

    @pytest.mark.unit
    def test_adds_component_labels(self):
        """Conversion adds componentLabel field."""
        html_hierarchy = {"tag": "button", "class": "html.button", "children": []}
        result = html_hierarchy_to_layout_hierarchy(html_hierarchy)

        assert "componentLabel" in result
        assert result["componentLabel"] == "Text Button"

    @pytest.mark.unit
    def test_infers_button_from_class(self):
        """Button-like classes are detected."""
        html_hierarchy = {
            "tag": "a",
            "class": "html.a",
            "html_class": "btn btn-primary",
            "children": [],
        }
        result = html_hierarchy_to_layout_hierarchy(html_hierarchy)

        assert result["componentLabel"] == "Text Button"

    @pytest.mark.unit
    def test_infers_card_from_class(self):
        """Card-like classes are detected."""
        html_hierarchy = {
            "tag": "div",
            "class": "html.div",
            "html_class": "card shadow",
            "children": [],
        }
        result = html_hierarchy_to_layout_hierarchy(html_hierarchy)

        assert result["componentLabel"] == "Card"

    @pytest.mark.unit
    def test_input_types_mapped(self):
        """Different input types map to correct components."""
        test_cases = [
            ({"tag": "input", "input_type": "checkbox"}, "Checkbox"),
            ({"tag": "input", "input_type": "radio"}, "Radio Button"),
            ({"tag": "input", "input_type": "submit"}, "Text Button"),
            ({"tag": "input", "input_type": "text"}, "Input"),
            ({"tag": "input", "input_type": "range"}, "Slider"),
        ]

        for html_node, expected_label in test_cases:
            html_node["class"] = f"html.{html_node['tag']}"
            html_node["children"] = []
            result = html_hierarchy_to_layout_hierarchy(html_node)
            assert result["componentLabel"] == expected_label, f"Failed for {html_node}"


class TestNormalizeHtmlToHierarchy:
    """Tests for the full HTML normalization pipeline."""

    @pytest.mark.unit
    def test_full_pipeline(self):
        """Full pipeline produces valid output."""
        html = """
        <html><body>
            <header><h1>Title</h1></header>
            <main>
                <p>Content</p>
                <button>Click</button>
            </main>
        </body></html>
        """
        hierarchy = normalize_html_to_hierarchy(html)

        assert "componentLabel" in hierarchy
        assert "children" in hierarchy
        assert len(hierarchy["children"]) > 0

    @pytest.mark.unit
    def test_produces_layout_compatible_output(self):
        """Output can be converted to LayoutNode."""
        html = "<div><p>Text</p><button>Click</button></div>"
        hierarchy = normalize_html_to_hierarchy(html)

        # Should be convertible to LayoutNode
        layout = hierarchy_to_layout(hierarchy)
        assert is_valid(layout)

    @pytest.mark.unit
    def test_handles_malformed_html(self):
        """Malformed HTML doesn't crash."""
        html = "<div><p>Unclosed<button>Also unclosed"
        hierarchy = normalize_html_to_hierarchy(html)

        # Should return something valid
        assert isinstance(hierarchy, dict)
        assert "children" in hierarchy

    @pytest.mark.unit
    def test_empty_html(self):
        """Empty HTML returns valid structure."""
        hierarchy = normalize_html_to_hierarchy("")

        assert isinstance(hierarchy, dict)
        assert "children" in hierarchy


class TestHtmlTagMap:
    """Tests for the HTML tag mapping."""

    @pytest.mark.unit
    def test_all_values_are_component_types(self):
        """All mapped values are valid ComponentTypes."""
        for tag, comp_type in HTML_TAG_MAP.items():
            assert isinstance(comp_type, ComponentType), f"Invalid mapping for {tag}"

    @pytest.mark.unit
    def test_common_tags_mapped(self):
        """Common HTML tags are in the mapping."""
        common_tags = ["div", "button", "input", "img", "p", "h1", "a", "nav", "header"]
        for tag in common_tags:
            assert tag in HTML_TAG_MAP, f"Missing mapping for {tag}"


class TestEndToEndHtmlNormalization:
    """End-to-end tests with realistic HTML."""

    @pytest.mark.unit
    def test_landing_page_html(self):
        """Landing page HTML normalizes correctly."""
        html = """
        <html><body>
            <nav class="navbar">
                <a href="/" class="logo">Brand</a>
                <button class="menu-btn">Menu</button>
            </nav>
            <main>
                <section class="hero">
                    <h1>Welcome</h1>
                    <p>Subtitle text</p>
                    <a href="#" class="btn cta">Get Started</a>
                </section>
            </main>
            <footer>
                <p>Copyright 2024</p>
            </footer>
        </body></html>
        """
        hierarchy = normalize_html_to_hierarchy(html)
        layout = hierarchy_to_layout(hierarchy)

        assert is_valid(layout)
        # Should have nav, main, footer as children
        assert len(layout.children) >= 3

    @pytest.mark.unit
    def test_form_html(self):
        """Form HTML normalizes correctly."""
        html = """
        <form>
            <label>Name</label>
            <input type="text" placeholder="Enter name">
            <label>Email</label>
            <input type="email" placeholder="Enter email">
            <input type="checkbox"> Subscribe
            <button type="submit">Submit</button>
        </form>
        """
        hierarchy = normalize_html_to_hierarchy(html)
        layout = hierarchy_to_layout(hierarchy)

        assert is_valid(layout)
