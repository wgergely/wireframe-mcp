"""VectorStore - Main orchestrator for vector database operations.

Provides high-level API for indexing corpus data and searching
for similar UI layouts using vector embeddings.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from src.corpus.provider.base import BaseProvider, StandardizedData
from src.mid import LayoutNode

from .backend import EmbeddingBackend, LocalBackend, VoyageBackend
from .index import FAISSIndex
from .serializer import LayoutSerializer, SerializationConfig, SerializedLayout

logger = logging.getLogger(__name__)

# Default batch size for embedding API calls
DEFAULT_BATCH_SIZE = 100


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

    Example:
        >>> store = VectorStore(backend="voyage")
        >>> store.index_corpus(provider, progress_callback=print)
        >>> results = store.search("login form with social auth", k=5)
    """

    def __init__(
        self,
        backend: EmbeddingBackend | Literal["voyage", "local"] = "voyage",
        index_path: Path | None = None,
        serializer_config: SerializationConfig | None = None,
    ):
        """Initialize VectorStore.

        Args:
            backend: Embedding backend instance or name ('voyage', 'local').
            index_path: Path for persistent index storage.
            serializer_config: Configuration for layout serialization.

        Raises:
            ValueError: If unknown backend name provided.
        """
        # Initialize backend
        if isinstance(backend, str):
            self._backend = self._create_backend(backend)
        else:
            self._backend = backend

        # Initialize serializer
        self._serializer = LayoutSerializer(serializer_config)

        # Initialize or load index
        self._index_path = index_path
        self._index: FAISSIndex | None = None
        self._metadata: dict[str, dict] = {}  # id -> metadata

        if index_path and Path(index_path).with_suffix(".faiss").exists():
            self.load(index_path)
        else:
            self._index = FAISSIndex(dimension=self._backend.dimension)

    def _create_backend(self, name: str) -> EmbeddingBackend:
        """Create embedding backend by name.

        Args:
            name: Backend name ('voyage', 'local').

        Returns:
            Initialized EmbeddingBackend instance.

        Raises:
            ValueError: If unknown backend name.
        """
        if name == "voyage":
            return VoyageBackend()
        elif name == "local":
            return LocalBackend()
        else:
            raise ValueError(
                f"Unknown backend '{name}'. Use 'voyage' or 'local', "
                "or pass an EmbeddingBackend instance."
            )

    def index_corpus(
        self,
        provider: BaseProvider,
        batch_size: int = DEFAULT_BATCH_SIZE,
        workers: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """Index all data from a corpus provider.

        Args:
            provider: Corpus provider to index.
            batch_size: Items per embedding batch.
            workers: Threads for serialization.
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

        # Serialize all items
        serialized: list[SerializedLayout] = []
        for i, item in enumerate(items):
            result = self._serializer.serialize_with_metadata(
                item.layout,
                item_id=item.id,
                source=item.source,
                dataset=item.dataset,
            )
            serialized.append(result)

            # Store metadata for retrieval
            self._metadata[item.id] = {
                "source": item.source,
                "dataset": item.dataset,
                "text": result.text,
                "node_count": result.node_count,
                "max_depth": result.max_depth,
                "component_summary": result.component_summary,
            }

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, total)

        logger.info(f"Serialized {len(serialized)} items, starting embedding...")

        # Embed and index in batches
        indexed = 0
        for i in range(0, len(serialized), batch_size):
            batch = serialized[i : i + batch_size]
            texts = [s.text for s in batch]
            ids = [s.id for s in batch]

            # Generate embeddings
            vectors = self._backend.embed(texts)

            # Add to index
            self._index.add(vectors, ids)
            indexed += len(batch)

            if progress_callback:
                progress_callback(indexed, total)

            logger.debug(f"Indexed batch {i // batch_size + 1}: {len(batch)} items")

        # Calculate source counts
        sources: dict[str, int] = {}
        for item in items:
            sources[item.source] = sources.get(item.source, 0) + 1

        stats = IndexStats(
            total_items=self._index.size,
            dimension=self._backend.dimension,
            embedding_backend=self._backend.name,
            is_gpu=self._index.is_gpu,
            sources=sources,
        )

        logger.info(
            f"Indexing complete: {stats.total_items} items, "
            f"GPU={stats.is_gpu}, backend={stats.embedding_backend}"
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

        # Store metadata
        self._metadata[data.id] = {
            "source": data.source,
            "dataset": data.dataset,
            "text": result.text,
            "node_count": result.node_count,
            "max_depth": result.max_depth,
            "component_summary": result.component_summary,
        }

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
            meta = self._metadata.get(raw.id, {})

            # Apply source filter
            if source_filter and meta.get("source") != source_filter:
                continue

            results.append(
                VectorSearchResult(
                    id=raw.id,
                    score=raw.score,
                    rank=len(results),
                    source=meta.get("source", "unknown"),
                    dataset=meta.get("dataset", "unknown"),
                    serialized_text=meta.get("text"),
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
            path: Save path (uses constructor path if None).

        Raises:
            ValueError: If no path available.
        """
        path = path or self._index_path
        if path is None:
            raise ValueError("No save path specified")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self._index.save(path)

        # Save metadata
        meta_path = path.with_suffix(".store.json")
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
            path: Load path.

        Raises:
            FileNotFoundError: If files not found.
        """
        path = Path(path)

        # Load FAISS index
        self._index = FAISSIndex.load(path)

        # Load metadata
        meta_path = path.with_suffix(".store.json")
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
                self._metadata = data.get("metadata", {})

        self._index_path = path
        logger.info(f"Loaded VectorStore from {path}")

    def stats(self) -> IndexStats:
        """Get current index statistics.

        Returns:
            IndexStats with current state.
        """
        sources: dict[str, int] = {}
        for meta in self._metadata.values():
            source = meta.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1

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


__all__ = [
    "VectorStore",
    "IndexStats",
    "VectorSearchResult",
]
