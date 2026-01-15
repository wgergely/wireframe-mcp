"""Audit script for hierarchy to LayoutNode normalization."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.corpus import CorpusManager
from src.corpus.normalizer import (
    count_components,
    extract_text_content,
    node_count,
    normalize_enrico_hierarchy,
    normalize_rico_hierarchy,
    tree_depth,
)
from src.mid import validate_layout


def audit_normalization():
    """Audit the normalization pipeline with real data."""
    print("=" * 60)
    print("NORMALIZATION AUDIT")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    manager = CorpusManager(data_dir=data_dir)

    # Test Rico normalization
    print("\n[Rico Normalization]")
    print("-" * 40)

    rico_items = []
    for i, item in enumerate(manager.stream_data("rico_semantic")):
        if i >= 20:
            break
        rico_items.append(item)

    rico_results = {
        "total": len(rico_items),
        "valid": 0,
        "invalid": 0,
        "total_nodes": 0,
        "max_depth": 0,
        "component_counts": {},
    }

    for item in rico_items:
        try:
            layout = normalize_rico_hierarchy(item.hierarchy, item.id)
            errors = validate_layout(layout)

            if not errors:
                rico_results["valid"] += 1
            else:
                rico_results["invalid"] += 1
                print(f"  Validation errors for {item.id}: {errors[:2]}")

            rico_results["total_nodes"] += node_count(layout)
            depth = tree_depth(layout)
            rico_results["max_depth"] = max(rico_results["max_depth"], depth)

            for comp_type, count in count_components(layout).items():
                rico_results["component_counts"][comp_type] = (
                    rico_results["component_counts"].get(comp_type, 0) + count
                )

        except Exception as e:
            print(f"  Error normalizing {item.id}: {e}")
            rico_results["invalid"] += 1

    print(f"  Total items: {rico_results['total']}")
    print(f"  Valid layouts: {rico_results['valid']}")
    print(f"  Invalid layouts: {rico_results['invalid']}")
    print(f"  Total nodes: {rico_results['total_nodes']}")
    print(
        f"  Avg nodes/item: {rico_results['total_nodes'] / rico_results['total']:.1f}"
    )
    print(f"  Max tree depth: {rico_results['max_depth']}")
    print("  Component distribution:")
    for comp, count in sorted(
        rico_results["component_counts"].items(), key=lambda x: -x[1]
    )[:10]:
        print(f"    {comp}: {count}")

    # Test Enrico normalization
    print("\n[Enrico Normalization]")
    print("-" * 40)

    enrico_items = []
    try:
        for i, item in enumerate(manager.stream_data("enrico")):
            if i >= 20:
                break
            enrico_items.append(item)
    except FileNotFoundError:
        print("  Enrico data not available, skipping...")
        enrico_items = []

    if enrico_items:
        enrico_results = {
            "total": len(enrico_items),
            "valid": 0,
            "invalid": 0,
            "total_nodes": 0,
            "max_depth": 0,
            "component_counts": {},
        }

        for item in enrico_items:
            try:
                layout = normalize_enrico_hierarchy(item.hierarchy, item.id)
                errors = validate_layout(layout)

                if not errors:
                    enrico_results["valid"] += 1
                else:
                    enrico_results["invalid"] += 1

                enrico_results["total_nodes"] += node_count(layout)
                enrico_results["max_depth"] = max(
                    enrico_results["max_depth"], tree_depth(layout)
                )

                for comp_type, count in count_components(layout).items():
                    enrico_results["component_counts"][comp_type] = (
                        enrico_results["component_counts"].get(comp_type, 0) + count
                    )

            except Exception as e:
                print(f"  Error normalizing {item.id}: {e}")
                enrico_results["invalid"] += 1

        print(f"  Total items: {enrico_results['total']}")
        print(f"  Valid layouts: {enrico_results['valid']}")
        print(f"  Invalid layouts: {enrico_results['invalid']}")
        print(f"  Total nodes: {enrico_results['total_nodes']}")
        avg_nodes = enrico_results['total_nodes'] / enrico_results['total']
        print(f"  Avg nodes/item: {avg_nodes:.1f}")
        print(f"  Max tree depth: {enrico_results['max_depth']}")
        print("  Component distribution:")
        for comp, count in sorted(
            enrico_results["component_counts"].items(), key=lambda x: -x[1]
        )[:10]:
            print(f"    {comp}: {count}")

    # Show sample normalized output
    print("\n[Sample Normalized Output]")
    print("-" * 40)

    if rico_items:
        sample = rico_items[0]
        layout = normalize_rico_hierarchy(sample.hierarchy, sample.id)
        print(f"  Source: {sample.source}, ID: {sample.id}")
        print(f"  Root type: {layout.type}")
        print(f"  Root orientation: {layout.orientation}")
        print(f"  Children count: {len(layout.children)}")
        print(f"  Total nodes: {node_count(layout)}")
        print(f"  Tree depth: {tree_depth(layout)}")
        print(f"  Text content: {extract_text_content(layout)[:5]}")

        # Print first level of tree
        print("\n  Tree structure (depth 2):")
        root_type = layout.type.value if hasattr(layout.type, "value") else layout.type
        print(f"    {root_type} [{layout.id}]")
        for child in layout.children[:5]:
            child_type = (
                child.type.value if hasattr(child.type, "value") else child.type
            )
            label_preview = f' "{child.label[:20]}..."' if child.label else ""
            print(f"      - {child_type}{label_preview} [{child.id}]")
            for grandchild in child.children[:3]:
                gc_type = (
                    grandchild.type.value
                    if hasattr(grandchild.type, "value")
                    else grandchild.type
                )
                gc_label = f' "{grandchild.label[:15]}..."' if grandchild.label else ""
                print(f"          - {gc_type}{gc_label}")
            if len(child.children) > 3:
                print(f"          ... +{len(child.children) - 3} more")
        if len(layout.children) > 5:
            print(f"      ... +{len(layout.children) - 5} more children")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    audit_normalization()
