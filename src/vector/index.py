"""FAISS vector index wrapper with GPU acceleration.

Provides a high-level interface for vector similarity search using FAISS.
Supports automatic GPU/CPU fallback and persistent storage.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a vector similarity search.

    Attributes:
        id: Original item identifier.
        score: Similarity score (higher is more similar).
        rank: Position in results (0-indexed).
    """

    id: str
    score: float
    rank: int


class FAISSIndex:
    """FAISS-based vector index with GPU acceleration.

    Provides efficient similarity search using FAISS. Automatically
    detects GPU availability and falls back to CPU if needed.

    Features:
        - GPU acceleration with automatic CPU fallback
        - Cosine similarity via normalized inner product
        - Persistent storage (save/load)
        - ID mapping for result lookup

    Example:
        >>> index = FAISSIndex(dimension=1024)
        >>> index.add(vectors, ids)
        >>> results = index.search(query_vector, k=5)
    """

    def __init__(
        self,
        dimension: int,
        use_gpu: bool | None = None,
    ):
        """Initialize FAISS index.

        Args:
            dimension: Vector dimension size.
            use_gpu: Force GPU (True), CPU (False), or auto-detect (None).
        """
        self._dimension = dimension
        self._index: Any = None
        self._id_map: list[str] = []
        self._is_gpu = False

        # Determine GPU usage
        if use_gpu is None:
            # Check environment override
            env_gpu = os.environ.get("VECTOR_USE_GPU", "").lower()
            if env_gpu == "false":
                use_gpu = False
            elif env_gpu == "true":
                use_gpu = True
            else:
                use_gpu = self._detect_gpu()
        elif use_gpu:
            use_gpu = self._detect_gpu()

        self._init_index(use_gpu)

    def _detect_gpu(self) -> bool:
        """Detect if FAISS GPU is available.

        Returns:
            True if GPU available, False otherwise.
        """
        try:
            import faiss

            if hasattr(faiss, "get_num_gpus"):
                return faiss.get_num_gpus() > 0
            return False
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False

    def _init_index(self, use_gpu: bool) -> None:
        """Initialize the FAISS index.

        Args:
            use_gpu: Whether to use GPU index.
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "FAISS required for vector indexing. "
                "Install with: pip install faiss-cpu or faiss-gpu"
            ) from e

        # Create CPU index first
        cpu_index = faiss.IndexFlatIP(self._dimension)

        if use_gpu:
            try:
                # Try to create GPU index
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self._is_gpu = True
                logger.info(f"Initialized FAISS GPU index (dim={self._dimension})")
            except Exception as e:
                logger.warning(f"GPU index failed, falling back to CPU: {e}")
                self._index = cpu_index
                self._is_gpu = False
                logger.info(f"Initialized FAISS CPU index (dim={self._dimension})")
        else:
            self._index = cpu_index
            self._is_gpu = False
            logger.info(f"Initialized FAISS CPU index (dim={self._dimension})")

    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension

    @property
    def size(self) -> int:
        """Get number of indexed vectors."""
        return len(self._id_map)

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU index."""
        return self._is_gpu

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Add vectors to the index.

        Vectors are normalized before adding for cosine similarity.

        Args:
            vectors: NumPy array of shape (n, dimension).
            ids: List of n string identifiers.

        Raises:
            ValueError: If vectors and ids length mismatch.
        """
        if len(vectors) != len(ids):
            raise ValueError(
                f"Vector count ({len(vectors)}) must match ID count ({len(ids)})"
            )

        if len(vectors) == 0:
            return

        # Ensure correct shape and type
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) "
                f"must match index dimension ({self._dimension})"
            )

        # Normalize for cosine similarity (inner product on normalized = cosine)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        vectors = vectors / norms

        # Add to index
        self._index.add(vectors)
        self._id_map.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query: Query vector of shape (dimension,) or (1, dimension).
            k: Number of results to return.

        Returns:
            List of SearchResult sorted by similarity (highest first).
        """
        if self.size == 0:
            return []

        # Ensure correct shape
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Limit k to index size
        k = min(k, self.size)

        # Search
        scores, indices = self._index.search(query, k)

        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True)):
            if idx >= 0 and idx < len(self._id_map):
                results.append(
                    SearchResult(
                        id=self._id_map[idx],
                        score=float(score),
                        rank=rank,
                    )
                )

        return results

    def save(self, path: Path) -> None:
        """Save index to disk.

        Creates two files:
        - {path}.faiss: The FAISS index
        - {path}.meta.json: ID mapping and metadata

        Args:
            path: Base path for save files (without extension).
        """
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert GPU index to CPU for saving
        if self._is_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self._index)
        else:
            cpu_index = self._index

        # Save FAISS index
        faiss.write_index(cpu_index, str(path.with_suffix(".faiss")))

        # Save metadata
        meta = {
            "dimension": self._dimension,
            "size": len(self._id_map),
            "is_gpu": self._is_gpu,
            "id_map": self._id_map,
        }
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)

        logger.info(f"Saved index to {path} ({self.size} vectors)")

    @classmethod
    def load(cls, path: Path, use_gpu: bool | None = None) -> "FAISSIndex":
        """Load index from disk.

        Args:
            path: Base path for load files (without extension).
            use_gpu: Force GPU (True), CPU (False), or auto-detect (None).

        Returns:
            Loaded FAISSIndex instance.

        Raises:
            FileNotFoundError: If index files not found.
        """
        import faiss

        path = Path(path)

        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Index metadata not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        # Load FAISS index
        faiss_path = path.with_suffix(".faiss")
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

        cpu_index = faiss.read_index(str(faiss_path))

        # Create instance
        instance = cls.__new__(cls)
        instance._dimension = meta["dimension"]
        instance._id_map = meta["id_map"]

        # Determine GPU usage
        if use_gpu is None:
            env_gpu = os.environ.get("VECTOR_USE_GPU", "").lower()
            if env_gpu == "false":
                use_gpu = False
            elif env_gpu == "true":
                use_gpu = instance._detect_gpu()
            else:
                use_gpu = instance._detect_gpu()

        # Move to GPU if requested
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                instance._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                instance._is_gpu = True
            except Exception as e:
                logger.warning(f"GPU load failed, using CPU: {e}")
                instance._index = cpu_index
                instance._is_gpu = False
        else:
            instance._index = cpu_index
            instance._is_gpu = False

        logger.info(
            f"Loaded index from {path} ({instance.size} vectors, "
            f"GPU={instance._is_gpu})"
        )
        return instance

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._init_index(self._is_gpu)
        self._id_map = []


__all__ = ["FAISSIndex", "SearchResult"]
