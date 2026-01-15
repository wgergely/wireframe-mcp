"""Ad-hoc audit script for ShowUI provider pipeline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.corpus import CorpusManager


def audit_showui_pipeline():
    """Audit the ShowUI provider pipeline step by step."""
    print("=" * 60)
    print("SHOWUI PROVIDER AUDIT")
    print("=" * 60)

    # Step 1: Initialize CorpusManager
    print("\n[Step 1] Initializing CorpusManager...")
    data_dir = Path(__file__).parent.parent / "data"
    manager = CorpusManager(data_dir=data_dir)
    print(f"  Data directory: {manager.data_dir}")

    # Step 2: Get ShowUI provider
    print("\n[Step 2] Getting showui provider...")
    try:
        provider = manager.get_provider("showui")
        print(f"  Provider name: {provider.name}")
        print(f"  Samples dir: {provider._samples_dir}")
    except KeyError as e:
        print(f"  ERROR: {e}")
        return False

    # Step 3: Check if data exists, fetch if needed
    print("\n[Step 3] Checking/fetching data...")
    if not provider._has_data():
        print("  Data not found, attempting download...")
        try:
            manager.fetch_dataset("showui")
            print("  Download complete!")
        except Exception as e:
            print(f"  ERROR during fetch: {e}")
            print("  NOTE: ShowUI requires 'datasets' library.")
            print("  Install with: pip install datasets")
            print(f"  Or manually place JSON samples in: {provider._samples_dir}")
            return False
    else:
        print(f"  Data exists at {provider._dest_dir}")

    # Count files
    json_files = list(provider._samples_dir.glob("*.json"))
    print(f"  JSON sample files: {len(json_files)}")

    # Step 4: Test data streaming
    print("\n[Step 4] Testing data streaming...")
    items = []
    sample_size = 100

    try:
        for i, item in enumerate(manager.stream_data("showui")):
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
        "has_source_showui": 0,
        "has_dataset_desktop": 0,
        "has_hierarchy": 0,
        "hierarchy_is_flat": 0,
        "has_instruction": 0,
        "has_detections": 0,
    }

    for item in items:
        if item.id:
            validation_results["has_id"] += 1
        if item.source == "showui":
            validation_results["has_source_showui"] += 1
        if item.dataset == "desktop":
            validation_results["has_dataset_desktop"] += 1
        if item.hierarchy:
            validation_results["has_hierarchy"] += 1
            # Check flat structure (depth=1)
            children = item.hierarchy.get("children", [])
            is_flat = all("children" not in c for c in children)
            if is_flat:
                validation_results["hierarchy_is_flat"] += 1
            # Check for instruction text
            if item.hierarchy.get("text"):
                validation_results["has_instruction"] += 1
            # Check for detections
            if len(children) > 0:
                validation_results["has_detections"] += 1

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
        print(f"  hierarchy type: {sample.hierarchy.get('type')}")
        children = sample.hierarchy.get("children", [])
        print(f"  detection count: {len(children)}")
        if children:
            print(f"  first detection: {children[0]}")
        print(f"  metadata: {sample.metadata}")

    # Step 7: Analyze detection types
    print("\n[Step 7] Analyzing detections...")
    total_detections = 0
    label_counts: dict[str, int] = {}

    for item in items:
        children = item.hierarchy.get("children", [])
        total_detections += len(children)
        for child in children:
            label = child.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"  Total detections across samples: {total_detections}")
    print(f"  Average detections per item: {total_detections / len(items):.1f}")
    print(f"  Label distribution: {label_counts}")

    # Step 8: Analyze query types
    print("\n[Step 8] Analyzing query types...")
    query_counts: dict[str, int] = {}
    interface_counts: dict[str, int] = {}

    for item in items:
        qt = item.metadata.get("query_type")
        if qt:
            query_counts[qt] = query_counts.get(qt, 0) + 1
        iface = item.metadata.get("interfaces")
        if iface:
            interface_counts[iface] = interface_counts.get(iface, 0) + 1

    print(f"  Query type distribution: {query_counts}")
    print(f"  Interface distribution: {interface_counts}")

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    issues = []
    if validation_results["hierarchy_is_flat"] < len(items):
        issues.append("WARNING: Some hierarchies are not flat (unexpected)")
    if validation_results["has_detections"] < len(items) * 0.5:
        issues.append("WARNING: Less than 50% of items have detections")

    print("\nNOTES:")
    print("  - ShowUI has FLAT hierarchy (no tree structure)")
    print("  - Each sample is grounding-focused (instruction + bbox)")
    print("  - No screenshot files included in JSON samples")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] No major issues found")

    return len([i for i in issues if i.startswith("WARNING")]) == 0


if __name__ == "__main__":
    success = audit_showui_pipeline()
    sys.exit(0 if success else 1)
