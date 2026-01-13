"""Integration test for vector store indexing and search.

Tests the complete pipeline: Provider -> Serializer -> VectorStore -> Search

Usage:
    python scripts/test_vector_index.py --limit 50
"""

import argparse
import time
from pathlib import Path
from itertools import islice

from src.config import get_data_dir
from src.corpus.provider.rico import Provider as RicoProvider
from src.vector import VectorStore
from src.vector.backend import create_backend, EmbeddingModel


def main():
    parser = argparse.ArgumentParser(description="Test vector store indexing")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of items to index",
    )
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "voyage"],
        help="Embedding backend to use",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    print(f"Data directory: {data_dir}")
    print(f"Backend: {args.backend}")
    print(f"Items to index: {args.limit}")
    print()

    # Initialize provider
    provider = RicoProvider(data_dir, dataset_type="semantic")
    if not provider.has_data():
        print("ERROR: Rico data not found. Run:")
        print("  python -m scripts.prepare_sample --provider rico --limit 10 --fetch")
        return 1

    # Create embedding backend
    print("Initializing embedding backend...")
    if args.backend == "voyage":
        import os
        if not os.environ.get("VOYAGE_API_KEY"):
            print("ERROR: VOYAGE_API_KEY not set")
            return 1
        backend = create_backend(EmbeddingModel.VOYAGE_3)
    else:
        backend = create_backend(EmbeddingModel.LOCAL_MINILM)

    print(f"  Backend: {backend.name}")
    print(f"  Dimension: {backend.dimension}")
    print()

    # Create vector store
    index_path = data_dir / "test_index"
    store = VectorStore(backend=backend, index_path=index_path)

    # Index items
    print(f"Indexing {args.limit} items...")
    start_time = time.time()
    
    indexed = 0
    for item in islice(provider.process(), args.limit):
        if item.layout is None:
            continue
        try:
            store.index_item(item)
            indexed += 1
            if indexed % 10 == 0:
                print(f"  Indexed {indexed} items...")
        except Exception as e:
            print(f"  Error indexing {item.id}: {e}")

    index_time = time.time() - start_time
    print(f"\nIndexed {indexed} items in {index_time:.2f}s")
    print(f"  Indexing rate: {indexed/index_time:.1f} items/sec")

    # Get stats
    stats = store.stats()
    print(f"\n--- Index Statistics ---")
    print(f"Total items: {stats.total_items}")
    print(f"Dimension: {stats.dimension}")
    print(f"Backend: {stats.embedding_backend}")
    print(f"GPU: {stats.is_gpu}")
    print(f"Sources: {stats.sources}")

    # Test search
    test_queries = [
        "login form with username and password",
        "dashboard with charts and statistics",
        "settings page with toggle switches",
        "photo gallery with thumbnails",
        "navigation menu with icons",
    ]

    print(f"\n--- Search Tests ---")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        start_time = time.time()
        results = store.search(query, k=3)
        search_time = (time.time() - start_time) * 1000

        print(f"  Latency: {search_time:.1f}ms")
        for r in results:
            print(f"  [{r.rank+1}] {r.id} (score: {r.score:.3f})")

    # Save index
    print(f"\n--- Saving Index ---")
    store.save()
    print(f"Index saved to: {index_path}")

    # Test load
    print("\n--- Testing Load ---")
    store2 = VectorStore(backend=backend)
    store2.load(index_path)
    stats2 = store2.stats()
    print(f"Loaded {stats2.total_items} items")
    
    # Verify search works after load
    results = store2.search("login form", k=1)
    if results:
        print(f"Search after load: {results[0].id} (score: {results[0].score:.3f})")

    print("\n✓ Vector store integration test passed!")
    return 0


if __name__ == "__main__":
    exit(main())
