"""Corpus normalizer module."""

from .html import (
    HTML_TAG_MAP,
    html_hierarchy_to_layout_hierarchy,
    normalize_html_to_hierarchy,
    parse_html_to_hierarchy,
)
from .lib import (
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

__all__ = [
    # Rico/Enrico normalization
    "COMPONENT_LABEL_MAP",
    "ANDROID_CLASS_MAP",
    "hierarchy_to_layout",
    "normalize_rico_hierarchy",
    "normalize_enrico_hierarchy",
    # HTML normalization
    "HTML_TAG_MAP",
    "parse_html_to_hierarchy",
    "html_hierarchy_to_layout_hierarchy",
    "normalize_html_to_hierarchy",
    # Tree utilities
    "count_components",
    "extract_text_content",
    "tree_depth",
    "node_count",
]
