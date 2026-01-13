"""Sample preparation script for Phase 0 testing.

Fetches a small sample from corpus providers, normalizes to LayoutNode,
serializes for inspection, and tests the vector store indexing.

Usage:
    python scripts/prepare_sample.py --provider rico --limit 100
"""

import argparse
import json
from pathlib import Path
from itertools import islice

from src.config import get_data_dir
from src.corpus.provider.rico import Provider as RicoProvider
from src.vector.serializer import LayoutSerializer, SerializationConfig


def main():
    parser = argparse.ArgumentParser(description="Prepare sample data for testing")
    parser.add_argument(
        "--provider",
        default="rico",
        choices=["rico"],
        help="Corpus provider to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of items to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for serialized samples",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch data if not present",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    output_dir = args.output_dir or (data_dir / "samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Provider: {args.provider}")
    print(f"Limit: {args.limit}")
    print()

    # Initialize provider
    if args.provider == "rico":
        provider = RicoProvider(data_dir, dataset_type="semantic")
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    # Check if data exists
    if not provider.has_data():
        if args.fetch:
            print(f"Fetching {args.provider} data...")
            provider.fetch()
        else:
            print(f"ERROR: No data found for {args.provider}.")
            print(f"Run with --fetch to download, or manually run:")
            print(f"  python . corpus fetch --provider {args.provider}")
            return 1

    # Initialize serializer
    serializer = LayoutSerializer(
        config=SerializationConfig(
            include_category=True,
            include_orientation=True,
            include_flex=True,
        )
    )

    # Process samples
    print(f"\nProcessing up to {args.limit} items...")
    processed = 0
    samples_data = []

    for item in islice(provider.process(), args.limit):
        if item.layout is None:
            print(f"  Skipping {item.id}: no layout")
            continue

        # Serialize the layout
        serialized = serializer.serialize_with_metadata(
            node=item.layout,
            item_id=item.id,
            source=item.source,
            dataset=item.dataset,
        )

        samples_data.append({
            "id": serialized.id,
            "source": serialized.source,
            "dataset": serialized.dataset,
            "node_count": serialized.node_count,
            "max_depth": serialized.max_depth,
            "component_summary": serialized.component_summary,
            "text": serialized.text[:500] + "..." if len(serialized.text) > 500 else serialized.text,
        })

        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed} items...")

    print(f"\nProcessed {processed} items total")

    # Save sample summary
    summary_path = output_dir / f"{args.provider}_sample_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(samples_data, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    # Print sample statistics
    if samples_data:
        avg_nodes = sum(s["node_count"] for s in samples_data) / len(samples_data)
        avg_depth = sum(s["max_depth"] for s in samples_data) / len(samples_data)
        all_components = {}
        for s in samples_data:
            for comp, count in s["component_summary"].items():
                all_components[comp] = all_components.get(comp, 0) + count

        print(f"\n--- Sample Statistics ---")
        print(f"Total samples: {len(samples_data)}")
        print(f"Average nodes per layout: {avg_nodes:.1f}")
        print(f"Average tree depth: {avg_depth:.1f}")
        print(f"\nTop 10 component types:")
        for comp, count in sorted(all_components.items(), key=lambda x: -x[1])[:10]:
            print(f"  {comp}: {count}")

    # Show sample serialization
    if samples_data:
        print(f"\n--- Sample Serialization (first item) ---")
        print(samples_data[0]["text"])

    return 0


if __name__ == "__main__":
    exit(main())
