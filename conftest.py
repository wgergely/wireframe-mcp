"""Pytest configuration and fixtures.

Loads .env file for integration tests.
Provides session-scoped RAG index for E2E testing.
"""

from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default test index location
TEST_INDEX_PATH = Path("data/test-index")
TEST_INDEX_LIMIT = 50  # Small index for fast testing


@pytest.fixture(scope="session")
def rag_index_path(tmp_path_factory) -> Path:
    """Build or reuse a small RAG index for E2E testing.

    This fixture auto-builds a test index if one doesn't exist,
    using a small subset of corpus data for fast testing.

    Returns:
        Path to the test vector index.
    """
    from src.corpus.api import CorpusManager
    from src.vector import BackendType, VectorStore

    # Check if persistent test index exists
    if TEST_INDEX_PATH.with_suffix(".faiss").exists():
        return TEST_INDEX_PATH

    # Build in temp directory for session
    index_path = tmp_path_factory.mktemp("rag_index") / "test"

    # Initialize components
    manager = CorpusManager()

    # Try to get a provider with data
    provider = None
    for name in ["rico_semantic", "rico", "enrico"]:
        try:
            provider = manager.get_provider(name)
            if provider:
                break
        except Exception:
            continue

    if provider is None:
        pytest.skip("No corpus provider available for RAG index")

    # Build index with LOCAL backend (fast, no API key needed)
    store = VectorStore(backend=BackendType.LOCAL)

    # Index limited subset for speed
    items_indexed = 0
    for item in provider.process():
        if item.layout is not None:
            store.index_item(item)
            items_indexed += 1
            if items_indexed >= TEST_INDEX_LIMIT:
                break

    if items_indexed == 0:
        pytest.skip("No items available to build RAG index")

    store.save(index_path)
    return index_path


@pytest.fixture
def vector_store(rag_index_path):
    """Load the test vector store.

    Args:
        rag_index_path: Path to test index (from session fixture).

    Returns:
        Loaded VectorStore instance.
    """
    from src.vector import BackendType, VectorStore

    store = VectorStore(backend=BackendType.LOCAL)
    store.load(rag_index_path)
    return store
