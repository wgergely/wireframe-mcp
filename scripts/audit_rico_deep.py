"""Deep audit of Rico hierarchy structure."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.corpus import CorpusManager  # noqa: E402


def count_component_labels(node: dict, depth: int = 0) -> dict:
    """Recursively count componentLabel occurrences at each depth."""
    results = {
        "depth": depth,
        "has_label": "componentLabel" in node,
        "children_count": 0,
    }

    children_with_label = 0
    total_descendants_with_label = 0

    if "children" in node:
        results["children_count"] = len(node["children"])
        for child in node["children"]:
            if "componentLabel" in child:
                children_with_label += 1
            child_results = count_component_labels(child, depth + 1)
            total_descendants_with_label += child_results.get(
                "descendants_with_label", 0
            )
            if child_results.get("has_label"):
                total_descendants_with_label += 1

    results["children_with_label"] = children_with_label
    results["descendants_with_label"] = total_descendants_with_label
    return results


def audit_hierarchy_structure():
    """Audit the Rico hierarchy structure in detail."""
    print("=" * 60)
    print("RICO HIERARCHY STRUCTURE AUDIT")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    manager = CorpusManager(data_dir=data_dir)

    sample_size = 50
    items = []

    for i, item in enumerate(manager.stream_data("rico_semantic")):
        if i >= sample_size:
            break
        items.append(item)

    print(f"\nAnalyzing {len(items)} items...\n")

    # Analyze hierarchy structure
    root_has_label = 0
    total_labels_in_tree = 0
    max_depth = 0

    for item in items:
        hierarchy = item.hierarchy
        results = count_component_labels(hierarchy)

        if results["has_label"]:
            root_has_label += 1
        total_labels_in_tree += results["descendants_with_label"]
        if results["has_label"]:
            total_labels_in_tree += 1

    print("Hierarchy Analysis:")
    print(f"  Root nodes with componentLabel: {root_has_label}/{len(items)}")
    print(f"  Total componentLabels across all trees: {total_labels_in_tree}")
    print(f"  Average labels per tree: {total_labels_in_tree / len(items):.1f}")

    # Show sample hierarchy structure
    print("\n" + "-" * 60)
    print("Sample hierarchy structure (first item):")
    print("-" * 60)

    def print_tree(node: dict, indent: int = 0):
        prefix = "  " * indent
        label = node.get("componentLabel", "<no label>")
        cls = node.get("class", "<no class>").split(".")[-1]
        bounds = node.get("bounds", [])
        text = node.get("text", "")
        text_preview = f' text="{text[:20]}..."' if text else ""

        print(f"{prefix}- {cls}: {label}{text_preview}")

        if "children" in node and indent < 3:  # Limit depth for display
            for child in node["children"][:5]:  # Limit children for display
                print_tree(child, indent + 1)
            if len(node["children"]) > 5:
                print(f"{prefix}  ... and {len(node['children']) - 5} more children")

    if items:
        print_tree(items[0].hierarchy)

    # Analyze unique componentLabel values
    print("\n" + "-" * 60)
    print("Unique componentLabel values found:")
    print("-" * 60)

    def collect_labels(node: dict) -> set:
        labels = set()
        if "componentLabel" in node:
            labels.add(node["componentLabel"])
        if "children" in node:
            for child in node["children"]:
                labels.update(collect_labels(child))
        return labels

    all_labels = set()
    for item in items:
        all_labels.update(collect_labels(item.hierarchy))

    for label in sorted(all_labels):
        print(f"  - {label}")

    print(f"\nTotal unique labels: {len(all_labels)}")


if __name__ == "__main__":
    audit_hierarchy_structure()
