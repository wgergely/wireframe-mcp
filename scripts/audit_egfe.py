"""Ad-hoc audit script for EGFE provider pipeline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.corpus import CorpusManager  # noqa: E402


def collect_labels(node: dict) -> set:
    """Recursively collect all componentLabel or type values."""
    labels = set()
    if "componentLabel" in node:
        labels.add(node["componentLabel"])
    if "type" in node:
        labels.add(node["type"])
    if "children" in node:
        for child in node["children"]:
            labels.update(collect_labels(child))
    return labels


def audit_egfe_pipeline():
    """Audit the EGFE provider pipeline step by step."""
    print("=" * 60)
    print("EGFE PROVIDER AUDIT")
    print("=" * 60)

    # Step 1: Initialize CorpusManager
    print("\n[Step 1] Initializing CorpusManager...")
    data_dir = Path(__file__).parent.parent / "data"
    manager = CorpusManager(data_dir=data_dir)
    print(f"  Data directory: {manager.data_dir}")

    # Step 2: Get EGFE provider
    print("\n[Step 2] Getting egfe provider...")
    try:
        provider = manager.get_provider("egfe")
        print(f"  Provider name: {provider.name}")
        print(f"  Dest dir: {provider._dest_dir}")
    except KeyError as e:
        print(f"  ERROR: {e}")
        return False

    # Step 3: Check if data exists, fetch if needed
    print("\n[Step 3] Checking/fetching data...")
    if not provider._has_data():
        print("  Data not found, attempting download...")
        try:
            manager.fetch_dataset("egfe")
            print("  Download complete!")
        except Exception as e:
            print(f"  ERROR during fetch: {e}")
            print("  NOTE: If download fails, manually place EGFE data in:")
            print(f"        {provider._dest_dir}")
            return False
    else:
        print(f"  Data exists at {provider._dest_dir}")

    # Count files
    json_files = list(provider._dest_dir.rglob("*.json"))
    png_files = list(provider._dest_dir.rglob("*.png"))
    print(f"  JSON files: {len(json_files)}")
    print(f"  PNG files: {len(png_files)}")

    # Step 4: Test data streaming
    print("\n[Step 4] Testing data streaming...")
    items = []
    sample_size = 100

    try:
        for i, item in enumerate(manager.stream_data("egfe")):
            if i >= sample_size:
                break
            items.append(item)
        print(f"  Successfully streamed {len(items)} items")
    except Exception as e:
        print(f"  ERROR during streaming: {e}")
        return False

    if not items:
        print("  WARNING: No items streamed!")
        return False

    # Step 5: Validate StandardizedData fields
    print("\n[Step 5] Validating StandardizedData fields...")
    validation_results = {
        "has_id": 0,
        "has_source_egfe": 0,
        "has_dataset": 0,
        "has_hierarchy": 0,
        "has_layout": 0,
        "has_screenshot": 0,
        "hierarchy_has_type": 0,
        "hierarchy_has_name": 0,
    }

    for item in items:
        if item.id:
            validation_results["has_id"] += 1
        if item.source == "egfe":
            validation_results["has_source_egfe"] += 1
        if item.dataset:
            validation_results["has_dataset"] += 1
        if item.hierarchy:
            validation_results["has_hierarchy"] += 1
            if "type" in item.hierarchy:
                validation_results["hierarchy_has_type"] += 1
            if "name" in item.hierarchy:
                validation_results["hierarchy_has_name"] += 1
        if item.layout is not None:
            validation_results["has_layout"] += 1
        if item.screenshot_path is not None:
            validation_results["has_screenshot"] += 1

    print(f"  Sample size: {len(items)}")
    for key, count in validation_results.items():
        pct = (count / len(items)) * 100 if items else 0
        status = "[OK]" if pct >= 90 else "[WARN]" if pct >= 50 else "[FAIL]"
        print(f"  {status} {key}: {count}/{len(items)} ({pct:.1f}%)")

    # Step 6: Show sample item
    print("\n[Step 6] Sample StandardizedData item:")
    if items:
        sample = items[0]
        print(f"  id: {sample.id}")
        print(f"  source: {sample.source}")
        print(f"  dataset: {sample.dataset}")
        print(f"  screenshot_path: {sample.screenshot_path}")
        print(f"  layout: {sample.layout}")
        print(f"  hierarchy keys: {list(sample.hierarchy.keys())[:10]}")

    # Step 7: Analyze element types
    print("\n[Step 7] Analyzing element types...")
    all_labels = set()
    total_labels = 0
    for item in items:
        labels = collect_labels(item.hierarchy)
        all_labels.update(labels)
        total_labels += len(labels)

    print(f"  Unique types/labels found: {len(all_labels)}")
    print(f"  Average types per item: {total_labels / len(items):.1f}")
    if all_labels:
        sample_labels = sorted(all_labels)[:20]
        print(f"  Sample types: {sample_labels}")

    # Step 8: Check screenshot files exist
    print("\n[Step 8] Validating screenshot file existence...")
    screenshots_found = 0
    screenshots_missing = 0
    for item in items:
        if item.screenshot_path:
            if item.screenshot_path.exists():
                screenshots_found += 1
            else:
                screenshots_missing += 1

    print(f"  Screenshots found: {screenshots_found}")
    print(f"  Screenshots missing: {screenshots_missing}")
    no_screenshot = len(items) - screenshots_found - screenshots_missing
    print(f"  Items without screenshot_path: {no_screenshot}")

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    issues = []
    if validation_results["has_layout"] == 0:
        issues.append("NOTE: layout field is not populated (expected for raw JSON)")
    if validation_results["has_screenshot"] < len(items) * 0.5:
        issues.append("WARNING: Less than 50% of items have screenshots")
    if screenshots_missing > 0:
        issues.append(
            f"WARNING: {screenshots_missing} screenshot paths reference missing files"
        )

    if issues:
        print("\nNotes/Warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] No major issues found")

    return len([i for i in issues if i.startswith("WARNING")]) == 0


if __name__ == "__main__":
    success = audit_egfe_pipeline()
    sys.exit(0 if success else 1)
