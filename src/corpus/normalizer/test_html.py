"""Unit tests for HTML normalizer."""

import pytest

from src.corpus.normalizer import hierarchy_to_layout
from src.corpus.normalizer.html import (
    HTML_TAG_MAP,
    html_hierarchy_to_layout_hierarchy,
    normalize_html_to_hierarchy,
    parse_html_to_hierarchy,
)
from src.mid import ComponentType, is_valid


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
        html = "<html><head><title>Test</title></head><body><div>Content</div></body></html>"
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
        html = '<input type="text" placeholder="Enter name" id="name-input" class="form-control">'
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
