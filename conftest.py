"""Root pytest configuration and fixtures.

This module provides:
- Environment setup (loads .env)
- Docker service management fixtures
- RAG index fixtures for E2E testing
- Global test configuration
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest
from dotenv import load_dotenv

from src.config import get_kroki_url

if TYPE_CHECKING:
    from src.mid import LayoutNode
    from src.render import RenderClient
    from src.vector import VectorStore

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Configuration Constants
# =============================================================================

TEST_INDEX_PATH = Path("data/test-index")
TEST_INDEX_LIMIT = 50  # Small index for fast testing
KROKI_URL = get_kroki_url()
DOCKER_COMPOSE_DIR = Path(__file__).parent / "docker"


# =============================================================================
# Docker Service Management (Private Functions)
# =============================================================================


def _is_docker_available() -> bool:
    """Check if Docker daemon is running."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _is_kroki_healthy(url: str = KROKI_URL, timeout: float = 2.0) -> bool:
    """Check if Kroki service is responding."""
    try:
        import httpx

        response = httpx.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _get_compose_files() -> list[Path]:
    """Get the list of compose files for Kroki service."""
    base = DOCKER_COMPOSE_DIR / "docker-compose.yml"
    kroki = DOCKER_COMPOSE_DIR / "docker-compose.kroki.yml"
    files = []
    if base.exists():
        files.append(base)
    if kroki.exists():
        files.append(kroki)
    return files


def _start_docker_service(service: str, compose_files: list[Path]) -> bool:
    """Start a Docker Compose service."""
    if not compose_files:
        return False

    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd.extend(["-f", str(f)])
    cmd.extend(["up", "-d", service])

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Modify test collection based on available services.

    Auto-skips tests marked with docker/kroki when services are unavailable.
    """
    docker_available = _is_docker_available()
    kroki_available = _is_kroki_healthy()

    skip_docker = pytest.mark.skip(reason="Docker not available")
    skip_kroki = pytest.mark.skip(reason="Kroki service not available")

    for item in items:
        # Skip docker tests if Docker unavailable
        if "docker" in item.keywords and not docker_available:
            item.add_marker(skip_docker)

        # Skip kroki tests if Kroki unavailable
        if "kroki" in item.keywords and not kroki_available:
            item.add_marker(skip_kroki)


# =============================================================================
# Docker Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Check if Docker is available for this test session.

    Returns:
        True if Docker daemon is running, False otherwise.
    """
    return _is_docker_available()


@pytest.fixture(scope="session")
def kroki_service(docker_available: bool) -> Generator[str, None, None]:
    """Ensure Kroki service is running.

    This fixture:
    1. Checks if Kroki is already running
    2. If not, attempts to start it via docker-compose (if Docker available)
    3. Waits for health check
    4. Yields the Kroki URL

    Services are left running after tests for faster re-runs.

    Yields:
        Kroki service URL.
    """
    # Check if already running
    if _is_kroki_healthy():
        yield KROKI_URL
        return

    # Need Docker to start service
    if not docker_available:
        pytest.skip("Docker not available and Kroki not running")

    # Get compose files
    compose_files = _get_compose_files()
    if not compose_files:
        pytest.skip("Docker compose files not found")

    # Start Kroki
    started = _start_docker_service("kroki", compose_files)
    if not started:
        pytest.skip("Failed to start Kroki service")

    # Wait for health check (max 60 seconds)
    for _ in range(60):
        if _is_kroki_healthy():
            break
        time.sleep(1)
    else:
        pytest.skip("Kroki service did not become healthy")

    yield KROKI_URL
    # Services left running for faster re-runs (per user preference)


@pytest.fixture(scope="function")
def kroki_client(kroki_service: str) -> RenderClient:
    """Create a RenderClient connected to Kroki.

    Args:
        kroki_service: Kroki URL from session fixture.

    Returns:
        Configured RenderClient instance.
    """
    from src.render import RenderClient

    client = RenderClient(base_url=kroki_service)
    if not client.is_available():
        pytest.skip("Kroki service not responding")
    return client


# =============================================================================
# RAG Index Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def rag_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
            if provider and provider.has_data():
                break
            provider = None
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
def vector_store(rag_index_path: Path) -> VectorStore:
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


# =============================================================================
# Common Test Fixtures
# =============================================================================


@pytest.fixture
def sample_layout() -> LayoutNode:
    """Create a simple sample LayoutNode for testing.

    Returns:
        A basic LayoutNode with header, content, and footer.
    """
    from src.mid import LayoutNode, Orientation
    from src.schema import ComponentType

    return LayoutNode(
        id="root",
        type=ComponentType.CONTAINER,
        orientation=Orientation.VERTICAL,
        children=[
            LayoutNode(id="header", type=ComponentType.TOOLBAR),
            LayoutNode(id="content", type=ComponentType.CONTAINER),
            LayoutNode(id="footer", type=ComponentType.NAVBAR),
        ],
    )


@pytest.fixture
def complex_layout() -> LayoutNode:
    """Create a complex nested LayoutNode for testing.

    Returns:
        A dashboard-style LayoutNode with sidebar and nested content.
    """
    from src.mid import LayoutNode, Orientation
    from src.schema import ComponentType

    return LayoutNode(
        id="dashboard",
        type=ComponentType.CONTAINER,
        orientation=Orientation.HORIZONTAL,
        children=[
            LayoutNode(
                id="sidebar",
                type=ComponentType.DRAWER,
                flex_ratio=3,
                children=[
                    LayoutNode(id="nav1", type=ComponentType.BUTTON, label="Home"),
                    LayoutNode(id="nav2", type=ComponentType.BUTTON, label="Settings"),
                ],
            ),
            LayoutNode(
                id="main",
                type=ComponentType.CONTAINER,
                flex_ratio=9,
                orientation=Orientation.VERTICAL,
                children=[
                    LayoutNode(id="toolbar", type=ComponentType.TOOLBAR),
                    LayoutNode(
                        id="content",
                        type=ComponentType.CARD,
                        children=[
                            LayoutNode(
                                id="title", type=ComponentType.TEXT, label="Dashboard"
                            ),
                            LayoutNode(id="chart", type=ComponentType.IMAGE),
                        ],
                    ),
                ],
            ),
        ],
    )
