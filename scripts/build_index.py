"""Build full corpus index.

Usage:
    python scripts/build_index.py --provider rico --backend local
"""

import argparse
import time
from itertools import islice

from src.config import get_data_dir
from src.corpus.provider.egfe import Provider as EGFEProvider
from src.corpus.provider.enrico import Provider as EnricoProvider
from src.corpus.provider.rico import Provider as RicoProvider
from src.vector import VectorStore
from src.vector.backend import EmbeddingModel, create_backend

PROVIDERS = {
    "rico": lambda d: RicoProvider(d, dataset_type="semantic"),
    "enrico": lambda d: EnricoProvider(d),
    "egfe": lambda d: EGFEProvider(d),
}


def main():
    parser = argparse.ArgumentParser(description="Build full corpus index")
    parser.add_argument(
        "--provider",
        default="rico",
        choices=list(PROVIDERS.keys()),
        help="Provider to index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit items (None for all)",
    )
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "voyage"],
        help="Embedding backend",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding (larger = faster on GPU)",
    )
    parser.add_argument(
        "--index-name",
        default="corpus_index",
        help="Index directory name",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    index_path = data_dir / args.index_name

    print(f"Provider: {args.provider}")
    print(f"Backend: {args.backend}")
    print(f"Index path: {index_path}")
    print(f"Limit: {args.limit or 'ALL'}")
    print()

    # Create provider
    provider = PROVIDERS[args.provider](data_dir)
    if not provider.has_data():
        print(f"ERROR: No data for {args.provider}. Run fetch first.")
        return 1

    # Create backend
    if args.backend == "voyage":
        import os

        if not os.environ.get("VOYAGE_API_KEY"):
            print("ERROR: VOYAGE_API_KEY not set")
            return 1
        backend = create_backend(EmbeddingModel.VOYAGE_3)
    else:
        backend = create_backend(EmbeddingModel.LOCAL_MINILM)

    print(f"Backend: {backend.name} ({backend.dimension}D)")

    # Create/load store
    store = VectorStore(backend=backend, index_path=index_path)
    if index_path.exists():
        try:
            store.load(index_path)
            print(f"Loaded existing index: {store.stats().total_items} items")
        except Exception:
            print("Starting fresh index")

    # Use batched index_corpus for GPU-optimized throughput
    print(f"\nIndexing {args.provider} (batch_size={args.batch_size})...")
    start = time.time()

    def progress(current, total):
        elapsed = time.time() - start
        rate = current / elapsed if elapsed > 0 else 0
        print(f"  {current}/{total} items ({rate:.1f}/sec)")

    # Wrap provider to apply limit
    class LimitedProvider:
        def __init__(self, provider, limit):
            self._provider = provider
            self._limit = limit

        @property
        def name(self):
            return self._provider.name

        def process(self):
            items = self._provider.process()
            if self._limit:
                items = islice(items, self._limit)
            return items

    limited = LimitedProvider(provider, args.limit)
    stats = store.index_corpus(
        limited,
        batch_size=args.batch_size,
        progress_callback=progress,
    )

    elapsed = time.time() - start
    print(f"\nIndexed {stats.total_items} items in {elapsed:.1f}s")
    print(f"Rate: {stats.total_items / elapsed:.1f} items/sec")
    print(f"GPU: {stats.is_gpu}")

    # Save
    store.save()
    print(f"\nIndex saved: {stats.total_items} total items")
    print(f"Sources: {stats.sources}")

    return 0


if __name__ == "__main__":
    exit(main())
