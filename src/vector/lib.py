"""VectorStore - Main orchestrator for vector database operations.

Provides high-level API for indexing corpus data and searching
for similar UI layouts using vector embeddings.
"""

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.corpus.provider.base import BaseProvider, StandardizedData
from src.mid import LayoutNode

from .backend import EmbeddingBackend, LocalBackend, VoyageBackend
from .index import FAISSIndex
from .serializer import LayoutSerializer, SerializationConfig, SerializedLayout
from .types import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_SERIALIZATION_WORKERS,
    UNKNOWN_DATASET,
    UNKNOWN_SOURCE,
    BackendType,
    ItemMetadata,
    MetadataKey,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics about the vector index.

    Attributes:
        total_items: Number of indexed items.
        dimension: Embedding vector dimension.
        embedding_backend: Backend name used for embeddings.
        is_gpu: Whether GPU acceleration is active.
        sources: Item counts by source provider.
    """

    total_items: int
    dimension: int
    embedding_backend: str
    is_gpu: bool
    sources: dict[str, int] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Extended search result with original data.

    Attributes:
        id: Item identifier.
        score: Similarity score (higher is better).
        rank: Position in results (0-indexed).
        source: Provider name.
        dataset: Dataset variant.
        layout: Original LayoutNode (if stored).
        serialized_text: Serialized text representation.
    """

    id: str
    score: float
    rank: int
    source: str
    dataset: str
    layout: LayoutNode | None = None
    serialized_text: str | None = None


class VectorStore:
    """Main API for vector database operations.

    Orchestrates serialization, embedding, and indexing of corpus data.
    Provides search interface for finding similar UI layouts.

    Features:
        - Parallel serialization with configurable worker threads
        - Embedding cache to skip duplicate content
        - GPU-accelerated FAISS indexing when available
        - Batched embedding generation

    Example:
        >>> store = VectorStore(backend="voyage")
        >>> store.index_corpus(provider, progress_callback=print)
        >>> results = store.search("login form with social auth", k=5)
    """

    def __init__(
        self,
        backend: EmbeddingBackend | BackendType | str = BackendType.VOYAGE,
        index_path: Path | None = None,
        serializer_config: SerializationConfig | None = None,
        enable_embedding_cache: bool = True,
    ):
        """Initialize VectorStore.

        Args:
            backend: Embedding backend instance, BackendType enum, or name string.
            index_path: Path for persistent index storage.
            serializer_config: Configuration for layout serialization.
            enable_embedding_cache: Cache embeddings to skip duplicates.

        Raises:
            ValueError: If unknown backend name provided.
        """
        # Initialize backend
        if isinstance(backend, (str, BackendType)):
            self._backend = self._create_backend(backend)
        else:
            self._backend = backend

        # Initialize serializer
        self._serializer = LayoutSerializer(serializer_config)

        # Initialize or load index
        self._index_path = index_path
        self._index: FAISSIndex | None = None
        self._metadata: dict[str, dict] = {}  # id -> metadata

        # Embedding cache for deduplication
        self._enable_cache = enable_embedding_cache
        self._embedding_cache: dict[str, np.ndarray] = {}

        if index_path and (Path(index_path) / "index.faiss").exists():
            self.load(index_path)
        else:
            self._index = FAISSIndex(dimension=self._backend.dimension)

    def _create_backend(self, backend: BackendType | str) -> EmbeddingBackend:
        """Create embedding backend by type.

        Args:
            backend: BackendType enum or name string.

        Returns:
            Initialized EmbeddingBackend instance.

        Raises:
            ValueError: If unknown backend type.
        """
        # Normalize to BackendType enum
        if isinstance(backend, str):
            try:
                backend = BackendType(backend)
            except ValueError:
                valid = [b.value for b in BackendType]
                raise ValueError(
                    f"Unknown backend '{backend}'. Valid options: {valid}"
                ) from None

        if backend == BackendType.VOYAGE:
            return VoyageBackend()
        elif backend == BackendType.LOCAL:
            return LocalBackend()
        else:
            # This shouldn't happen with enum, but satisfies type checker
            raise ValueError(f"Unhandled backend type: {backend}")

    def _get_text_hash(self, text: str) -> str:
        """Generate hash key for embedding cache.

        Args:
            text: Text to hash.

        Returns:
            MD5 hash string.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cached_embedding(self, text: str) -> np.ndarray | None:
        """Get cached embedding for text.

        Args:
            text: Text to look up.

        Returns:
            Cached embedding or None if not found.
        """
        if not self._enable_cache:
            return None
        key = self._get_text_hash(text)
        return self._embedding_cache.get(key)

    def _cache_embeddings(self, texts: list[str], vectors: np.ndarray) -> None:
        """Cache embeddings for texts.

        Creates explicit copies to prevent aliasing issues with mutable arrays.

        Args:
            texts: List of texts.
            vectors: Corresponding embedding vectors.
        """
        if not self._enable_cache:
            return
        for i, text in enumerate(texts):
            key = self._get_text_hash(text)
            # Explicit copy to prevent aliasing with source array
            self._embedding_cache[key] = vectors[i].copy()

    def index_corpus(
        self,
        provider: BaseProvider,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        workers: int = DEFAULT_SERIALIZATION_WORKERS,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """Index all data from a corpus provider.

        Uses parallel serialization and embedding cache for optimal performance.

        Args:
            provider: Corpus provider to index.
            batch_size: Items per embedding batch.
            workers: Threads for parallel serialization.
            progress_callback: Optional (current, total) progress callback.

        Returns:
            IndexStats with indexing statistics.
        """
        logger.info(f"Starting corpus indexing from provider '{provider.name}'")

        # Collect items for indexing
        items: list[StandardizedData] = []
        for item in provider.process():
            if item.layout is not None:
                items.append(item)

        total = len(items)
        logger.info(f"Found {total} items with layouts to index")

        if total == 0:
            return IndexStats(
                total_items=0,
                dimension=self._backend.dimension,
                embedding_backend=self._backend.name,
                is_gpu=self._index.is_gpu if self._index else False,
                sources={provider.name: 0},
            )

        # Parallel serialization with progress tracking
        serialized: list[SerializedLayout] = []
        serialized_count = 0

        def on_serialized(result: SerializedLayout) -> None:
            nonlocal serialized_count
            serialized_count += 1
            # Store metadata for retrieval (type-safe)
            self._metadata[result.id] = ItemMetadata(
                source=result.source,
                dataset=result.dataset,
                text=result.text,
                node_count=result.node_count,
                max_depth=result.max_depth,
                component_summary=result.component_summary,
            )
            if progress_callback and serialized_count % 100 == 0:
                # Report serialization progress (first half of work)
                progress_callback(serialized_count // 2, total)

        # Generate items for batch processing
        items_for_batch = (
            (item.layout, item.id, item.source, item.dataset) for item in items
        )

        # Use parallel serialization
        logger.info(f"Serializing {total} items with {workers} workers...")
        for result in self._serializer.serialize_batch(
            items_for_batch, max_workers=workers, on_complete=on_serialized
        ):
            serialized.append(result)

        logger.info(f"Serialized {len(serialized)} items, starting embedding...")

        # Embed and index in batches with caching
        indexed = 0
        cache_hits = 0

        for i in range(0, len(serialized), batch_size):
            batch = serialized[i : i + batch_size]

            # Separate cached vs uncached texts with intra-batch deduplication
            cached_indices: list[int] = []
            cached_vectors: list[np.ndarray] = []

            # Track unique uncached texts and which batch indices need them
            unique_texts: list[str] = []
            text_to_unique_idx: dict[str, int] = {}  # text -> index in unique_texts
            batch_idx_to_unique_idx: dict[int, int] = {}  # batch idx -> unique idx

            for j, s in enumerate(batch):
                cached = self._get_cached_embedding(s.text)
                if cached is not None:
                    cached_indices.append(j)
                    cached_vectors.append(cached)
                    cache_hits += 1
                else:
                    # Intra-batch deduplication: only add unique texts
                    if s.text not in text_to_unique_idx:
                        text_to_unique_idx[s.text] = len(unique_texts)
                        unique_texts.append(s.text)
                    batch_idx_to_unique_idx[j] = text_to_unique_idx[s.text]

            # Pre-allocate output array
            all_vectors = np.zeros(
                (len(batch), self._backend.dimension), dtype=np.float32
            )

            # Vectorized assignment of cached vectors
            if cached_vectors:
                all_vectors[cached_indices] = np.asarray(cached_vectors)

            # Generate embeddings only for unique uncached texts
            if unique_texts:
                new_vectors = self._backend.embed(unique_texts)
                self._cache_embeddings(unique_texts, new_vectors)
                # Map unique vectors back to all batch indices that need them
                for batch_idx, unique_idx in batch_idx_to_unique_idx.items():
                    all_vectors[batch_idx] = new_vectors[unique_idx]

            # Add to index
            ids = [s.id for s in batch]
            self._index.add(all_vectors, ids)
            indexed += len(batch)

            if progress_callback:
                # Report embedding progress (second half of work)
                progress_callback(total // 2 + indexed // 2, total)

            logger.debug(
                f"Indexed batch {i // batch_size + 1}: {len(batch)} items "
                f"({len(cached_indices)} cached)"
            )

        # Calculate source counts using Counter (more efficient)
        sources = dict(Counter(item.source for item in items))

        stats = IndexStats(
            total_items=self._index.size,
            dimension=self._backend.dimension,
            embedding_backend=self._backend.name,
            is_gpu=self._index.is_gpu,
            sources=sources,
        )

        logger.info(
            f"Indexing complete: {stats.total_items} items, "
            f"GPU={stats.is_gpu}, backend={stats.embedding_backend}, "
            f"cache_hits={cache_hits}"
        )

        return stats

    def index_item(self, data: StandardizedData) -> None:
        """Index a single item for incremental updates.

        Args:
            data: StandardizedData to index.

        Raises:
            ValueError: If item has no layout.
        """
        if data.layout is None:
            raise ValueError(f"Item {data.id} has no layout to index")

        # Serialize
        result = self._serializer.serialize_with_metadata(
            data.layout,
            item_id=data.id,
            source=data.source,
            dataset=data.dataset,
        )

        # Store metadata (type-safe)
        self._metadata[data.id] = ItemMetadata(
            source=data.source,
            dataset=data.dataset,
            text=result.text,
            node_count=result.node_count,
            max_depth=result.max_depth,
            component_summary=result.component_summary,
        )

        # Embed and add
        vectors = self._backend.embed([result.text])
        self._index.add(vectors, [data.id])

    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: str | None = None,
    ) -> list[VectorSearchResult]:
        """Find similar layouts for a natural language query.

        Args:
            query: Natural language query describing desired layout.
            k: Number of results to return.
            source_filter: Optional filter by source provider name.

        Returns:
            List of VectorSearchResult sorted by similarity.
        """
        if self._index.size == 0:
            logger.warning("Search on empty index")
            return []

        # Embed query
        query_vector = self._backend.embed_query(query)

        # Search with extra results if filtering
        search_k = k * 3 if source_filter else k
        raw_results = self._index.search(query_vector, k=search_k)

        # Build results with metadata
        results: list[VectorSearchResult] = []
        for raw in raw_results:
            meta = self._metadata.get(raw.id)

            # Apply source filter
            source = meta[MetadataKey.SOURCE] if meta else UNKNOWN_SOURCE
            if source_filter and source != source_filter:
                continue

            results.append(
                VectorSearchResult(
                    id=raw.id,
                    score=raw.score,
                    rank=len(results),
                    source=source,
                    dataset=meta[MetadataKey.DATASET] if meta else UNKNOWN_DATASET,
                    serialized_text=meta[MetadataKey.TEXT] if meta else None,
                )
            )

            if len(results) >= k:
                break

        return results

    def get_metadata(self, item_id: str) -> dict | None:
        """Get stored metadata for an item.

        Args:
            item_id: Item identifier.

        Returns:
            Metadata dict or None if not found.
        """
        return self._metadata.get(item_id)

    def save(self, path: Path | None = None) -> None:
        """Save index and metadata to disk.

        Args:
            path: Directory to save index files (uses constructor path if None).

        Raises:
            ValueError: If no path available.
        """
        path = path or self._index_path
        if path is None:
            raise ValueError("No save path specified")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index (saves to path/index.faiss and path/index.meta.json)
        self._index.save(path)

        # Save VectorStore metadata
        meta_path = path / "index.store.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "backend": self._backend.name,
                    "dimension": self._backend.dimension,
                    "metadata": self._metadata,
                },
                f,
            )

        logger.info(f"Saved VectorStore to {path}")

    def load(self, path: Path) -> None:
        """Load index and metadata from disk.

        Args:
            path: Directory containing index files.

        Raises:
            FileNotFoundError: If files not found.
        """
        path = Path(path)

        # Load metadata first to get backend type
        meta_path = path / "index.store.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
                self._metadata = data.get("metadata", {})

                # Restore backend from saved metadata
                saved_backend = data.get("backend", "")
                if saved_backend:
                    # Parse backend type (e.g., "local:all-MiniLM-L6-v2" -> "local")
                    backend_type = saved_backend.split(":")[0]
                    if backend_type in [b.value for b in BackendType]:
                        self._backend = self._create_backend(backend_type)
                        logger.info(f"Restored backend: {backend_type}")

        # Load FAISS index
        self._index = FAISSIndex.load(path)

        self._index_path = path
        logger.info(f"Loaded VectorStore from {path}")

    def stats(self) -> IndexStats:
        """Get current index statistics.

        Returns:
            IndexStats with current state.
        """
        # Use Counter for efficient source counting
        sources = dict(
            Counter(
                meta.get(MetadataKey.SOURCE, UNKNOWN_SOURCE)
                for meta in self._metadata.values()
            )
        )

        return IndexStats(
            total_items=self._index.size if self._index else 0,
            dimension=self._backend.dimension,
            embedding_backend=self._backend.name,
            is_gpu=self._index.is_gpu if self._index else False,
            sources=sources,
        )

    def clear(self) -> None:
        """Clear all indexed data."""
        if self._index:
            self._index.clear()
        self._metadata = {}

    def __len__(self) -> int:
        """Return number of indexed items."""
        return self._index.size if self._index else 0

    @property
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        return BackendType(self._backend.name)


__all__ = [
    "VectorStore",
    "IndexStats",
    "VectorSearchResult",
]
