"""Audit script for WebSight provider pipeline."""

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
    tree_depth,
)
from src.mid import is_valid


def audit_websight_pipeline():
    """Audit the WebSight provider pipeline."""
    print("=" * 60)
    print("WEBSIGHT PROVIDER AUDIT")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    manager = CorpusManager(data_dir=data_dir)

    # Step 1: Get provider
    print("\n[Step 1] Getting websight provider...")
    provider = manager.get_provider("websight")
    print(f"  Provider name: {provider.name}")
    print(f"  Data dir: {provider._dest_dir}")

    # Step 2: Fetch (creates sample data)
    print("\n[Step 2] Fetching data...")
    manager.fetch_dataset("websight")

    # Step 3: Process data
    print("\n[Step 3] Processing data...")
    items = list(manager.stream_data("websight"))
    print(f"  Items processed: {len(items)}")

    if not items:
        print("  ERROR: No items processed!")
        return False

    # Step 4: Validate items
    print("\n[Step 4] Validating StandardizedData...")
    results = {
        "has_id": 0,
        "has_source": 0,
        "has_hierarchy": 0,
        "has_layout": 0,
        "valid_layout": 0,
    }

    for item in items:
        if item.id:
            results["has_id"] += 1
        if item.source == "websight":
            results["has_source"] += 1
        if item.hierarchy:
            results["has_hierarchy"] += 1
        if item.layout:
            results["has_layout"] += 1
            if is_valid(item.layout):
                results["valid_layout"] += 1

    for key, count in results.items():
        pct = (count / len(items)) * 100
        status = "[OK]" if pct >= 90 else "[WARN]" if pct >= 50 else "[FAIL]"
        print(f"  {status} {key}: {count}/{len(items)} ({pct:.1f}%)")

    # Step 5: Analyze layouts
    print("\n[Step 5] Layout analysis...")
    total_nodes = 0
    total_depth = 0
    all_components = {}

    for item in items:
        if item.layout:
            total_nodes += node_count(item.layout)
            total_depth += tree_depth(item.layout)
            for comp, cnt in count_components(item.layout).items():
                all_components[comp] = all_components.get(comp, 0) + cnt

    print(f"  Avg nodes/item: {total_nodes / len(items):.1f}")
    print(f"  Avg depth: {total_depth / len(items):.1f}")
    print("  Component distribution:")
    for comp, cnt in sorted(all_components.items(), key=lambda x: -x[1])[:10]:
        print(f"    {comp}: {cnt}")

    # Step 6: Sample output
    print("\n[Step 6] Sample item...")
    sample = items[0]
    print(f"  ID: {sample.id}")
    print(f"  Source: {sample.source}")
    print(f"  Metadata: {sample.metadata}")
    print(f"  Hierarchy keys: {list(sample.hierarchy.keys())}")
    if sample.layout:
        print(f"  Layout root type: {sample.layout.type}")
        print(f"  Layout children: {len(sample.layout.children)}")
        texts = extract_text_content(sample.layout)[:5]
        print(f"  Text content: {texts}")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = audit_websight_pipeline()
    sys.exit(0 if success else 1)
