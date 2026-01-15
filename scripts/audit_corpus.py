"""Cross-provider corpus audit for vector DB embedding assessment.

This script examines and contrasts data from all corpus providers to assess:
1. Semantic normalization consistency across providers
2. Image data availability and usability
3. LayoutNode quality for vector embedding
4. Text content extraction for LLM grounding
"""

import sys
from collections import Counter
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
from src.mid import ComponentType, is_valid


def audit_provider(manager: CorpusManager, provider_name: str, limit: int = 50) -> dict:
    """Audit a single provider and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"AUDITING: {provider_name}")
    print("=" * 60)

    try:
        provider = manager.get_provider(provider_name)
    except KeyError:
        print("  [SKIP] Provider not found")
        return {"error": "not_found"}

    if not provider.has_data():
        print("  [SKIP] No data available")
        return {"error": "no_data"}

    metrics = {
        "provider": provider_name,
        "total": 0,
        "has_hierarchy": 0,
        "has_layout": 0,
        "layout_valid": 0,
        "has_screenshot": 0,
        "screenshot_exists": 0,
        "has_text_content": 0,
        "component_counts": Counter(),
        "avg_tree_depth": 0,
        "avg_node_count": 0,
        "avg_text_items": 0,
        "sample_layouts": [],
        "image_info": [],
    }

    depths = []
    node_counts = []
    text_counts = []

    try:
        for i, item in enumerate(manager.stream_data(provider_name)):
            if i >= limit:
                break

            metrics["total"] += 1

            # Hierarchy check
            if item.hierarchy:
                metrics["has_hierarchy"] += 1

            # Layout checks
            if item.layout is not None:
                metrics["has_layout"] += 1

                if is_valid(item.layout):
                    metrics["layout_valid"] += 1

                # Analyze layout structure
                depths.append(tree_depth(item.layout))
                node_counts.append(node_count(item.layout))

                comp_counts = count_components(item.layout)
                for comp_type, count in comp_counts.items():
                    metrics["component_counts"][comp_type] += count

                # Extract text
                texts = extract_text_content(item.layout)
                text_counts.append(len(texts))
                if texts:
                    metrics["has_text_content"] += 1

                # Store sample layouts (first 3)
                if len(metrics["sample_layouts"]) < 3:
                    metrics["sample_layouts"].append(
                        {
                            "id": item.id,
                            "depth": tree_depth(item.layout),
                            "nodes": node_count(item.layout),
                            "components": comp_counts,
                            "text_sample": texts[:3],
                        }
                    )

            # Screenshot checks
            if item.screenshot_path:
                metrics["has_screenshot"] += 1
                if item.screenshot_path.exists():
                    metrics["screenshot_exists"] += 1
                    # Get image info
                    if len(metrics["image_info"]) < 3:
                        stat = item.screenshot_path.stat()
                        metrics["image_info"].append(
                            {
                                "path": str(item.screenshot_path.name),
                                "size_kb": stat.st_size // 1024,
                                "suffix": item.screenshot_path.suffix,
                            }
                        )

    except Exception as e:
        print(f"  [ERROR] {e}")
        metrics["error"] = str(e)

    # Calculate averages
    if depths:
        metrics["avg_tree_depth"] = sum(depths) / len(depths)
    if node_counts:
        metrics["avg_node_count"] = sum(node_counts) / len(node_counts)
    if text_counts:
        metrics["avg_text_items"] = sum(text_counts) / len(text_counts)

    # Print summary
    total = metrics["total"]
    if total > 0:
        print(f"\n  Samples analyzed: {total}")
        hier_pct = 100 * metrics['has_hierarchy'] / total
        print(f"  Hierarchy: {metrics['has_hierarchy']}/{total} ({hier_pct:.0f}%)")
        layout_pct = 100 * metrics['has_layout'] / total
        print(f"  Layout: {metrics['has_layout']}/{total} ({layout_pct:.0f}%)")
        valid_pct = 100 * metrics['layout_valid'] / total
        print(f"  Layout valid: {metrics['layout_valid']}/{total} ({valid_pct:.0f}%)")
        scrn_pct = 100 * metrics['screenshot_exists'] / total
        sc_cnt = metrics['screenshot_exists']
        print(f"  Screenshots: {sc_cnt}/{total} ({scrn_pct:.0f}%)")
        text_pct = 100 * metrics['has_text_content'] / total
        print(f"  Has text: {metrics['has_text_content']}/{total} ({text_pct:.0f}%)")
        print(f"\n  Avg tree depth: {metrics['avg_tree_depth']:.1f}")
        print(f"  Avg node count: {metrics['avg_node_count']:.1f}")
        print(f"  Avg text items: {metrics['avg_text_items']:.1f}")

        if metrics["component_counts"]:
            print("\n  Top component types:")
            for comp, count in metrics["component_counts"].most_common(8):
                print(f"    {comp}: {count}")

        if metrics["image_info"]:
            print("\n  Sample images:")
            for img in metrics["image_info"]:
                print(f"    {img['path']} ({img['size_kb']}KB, {img['suffix']})")

    return metrics


def compare_providers(all_metrics: list[dict]) -> None:
    """Generate comparison report across providers."""
    print("\n" + "=" * 80)
    print("CROSS-PROVIDER COMPARISON")
    print("=" * 80)

    valid_metrics = [m for m in all_metrics if "error" not in m]

    if not valid_metrics:
        print("No valid provider data to compare")
        return

    # Comparison table
    print("\n### Data Quality Matrix\n")
    header = f"{'Provider':<15} {'Samples':>8} {'Layout%':>8} {'Valid%':>8} "
    header += f"{'Image%':>8} {'Text%':>8} {'Depth':>6} {'Nodes':>6}"
    print(header)
    print("-" * 80)

    for m in valid_metrics:
        total = m["total"]
        if total == 0:
            continue
        print(
            f"{m['provider']:<15} "
            f"{total:>8} "
            f"{100 * m['has_layout'] / total:>7.0f}% "
            f"{100 * m['layout_valid'] / total:>7.0f}% "
            f"{100 * m['screenshot_exists'] / total:>7.0f}% "
            f"{100 * m['has_text_content'] / total:>7.0f}% "
            f"{m['avg_tree_depth']:>6.1f} "
            f"{m['avg_node_count']:>6.1f}"
        )

    # Component type distribution
    print("\n### Component Type Distribution\n")
    all_components = Counter()
    for m in valid_metrics:
        all_components.update(m["component_counts"])

    known_types = set(ct.value for ct in ComponentType)
    print(f"{'Component Type':<20} {'Count':>8} {'MID Valid':>10}")
    print("-" * 40)
    for comp, count in all_components.most_common(15):
        in_mid = "✓" if comp in known_types else "✗"
        print(f"{comp:<20} {count:>8} {in_mid:>10}")

    # Assessment for Vector DB suitability
    print("\n### Vector DB Embedding Assessment\n")

    _total_layouts = sum(m["has_layout"] for m in valid_metrics)
    total_valid = sum(m["layout_valid"] for m in valid_metrics)
    total_text = sum(m["has_text_content"] for m in valid_metrics)
    total_images = sum(m["screenshot_exists"] for m in valid_metrics)
    total_samples = sum(m["total"] for m in valid_metrics)

    print(f"Total samples across providers: {total_samples}")
    valid_pct = 100 * total_valid / total_samples
    print(f"Total with valid LayoutNode: {total_valid} ({valid_pct:.0f}%)")
    text_pct = 100 * total_text / total_samples
    print(f"Total with text content: {total_text} ({text_pct:.0f}%)")
    print(
        f"Total with images: {total_images} ({100 * total_images / total_samples:.0f}%)"
    )

    print("\n--- Observations ---")

    if total_valid / total_samples > 0.8:
        print(
            "✓ HIGH: Most samples have valid LayoutNodes - good structural consistency"
        )
    elif total_valid / total_samples > 0.5:
        print("⚠ MEDIUM: Partial LayoutNode coverage - some normalization gaps")
    else:
        print(
            "✗ LOW: Limited LayoutNode coverage - significant normalization work needed"
        )

    if total_text / total_samples > 0.7:
        print("✓ TEXT: Rich text content available for semantic embeddings")
    else:
        print("⚠ TEXT: Limited text - may need image-based content extraction")

    if total_images / total_samples > 0.5:
        print("✓ IMAGES: Good image coverage for visual embeddings")
    else:
        print("⚠ IMAGES: Limited images - primarily structural/text embeddings")


def main():
    print("=" * 80)
    print("CORPUS CROSS-PROVIDER AUDIT")
    print("Assessing data quality for vector DB embedding and LLM grounding")
    print("=" * 80)

    data_dir = Path(__file__).parent.parent / "data"
    manager = CorpusManager(data_dir=data_dir)

    # List all providers
    providers = manager.list_providers()
    print(f"\nAvailable providers: {providers}")

    # Audit each provider
    all_metrics = []
    for provider_name in providers:
        metrics = audit_provider(manager, provider_name, limit=30)
        all_metrics.append(metrics)

    # Cross-provider comparison
    compare_providers(all_metrics)

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR VECTOR DB INTEGRATION")
    print("=" * 80)
    print("""
1. STRUCTURAL EMBEDDING (LayoutNode):
   - Serialize LayoutNode tree as JSON for structural embeddings
   - Use component type, depth, and node count as features
   - Consider tree-based embeddings (e.g., Tree-LSTM, GNN)

2. TEXT EMBEDDING:
   - Extract text labels from LayoutNode for semantic search
   - Combine with component context (e.g., "button: Submit")
   - Use sentence transformers for dense vectors

3. IMAGE EMBEDDING:
   - Use CLIP or similar for screenshot embeddings
   - Combined with structural features for multimodal retrieval
   - Consider cropping to component bounding boxes

4. HYBRID APPROACH:
   - Concatenate structural + text + image embeddings
   - Weight by availability (images sparse in some providers)
   - Store metadata for filtering (source, component types)
""")


if __name__ == "__main__":
    main()
