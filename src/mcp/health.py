"""Health checking for MCP server dependencies.

Provides centralized status checking for all server dependencies:
- Kroki rendering service
- RAG/vector index
- LLM provider availability
- History storage
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Overall health status."""

    HEALTHY = "healthy"  # All dependencies available
    DEGRADED = "degraded"  # Some features unavailable
    UNHEALTHY = "unhealthy"  # Critical dependencies missing


@dataclass
class ServiceStatus:
    """Status of a single service/dependency."""

    available: bool
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerHealth:
    """Complete server health report."""

    status: HealthStatus
    version: str
    checked_at: datetime

    # Service statuses
    kroki: ServiceStatus
    rag_index: ServiceStatus
    llm_providers: ServiceStatus
    history_db: ServiceStatus

    # Capability summary
    can_generate: bool  # LLM available
    can_preview: bool  # Kroki available
    can_use_rag: bool  # Index available

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "status": self.status.value,
            "version": self.version,
            "checked_at": self.checked_at.isoformat(),
            "services": {
                "kroki": {
                    "available": self.kroki.available,
                    "message": self.kroki.message,
                    **self.kroki.details,
                },
                "rag_index": {
                    "available": self.rag_index.available,
                    "message": self.rag_index.message,
                    **self.rag_index.details,
                },
                "llm_providers": {
                    "available": self.llm_providers.available,
                    "message": self.llm_providers.message,
                    **self.llm_providers.details,
                },
                "history_db": {
                    "available": self.history_db.available,
                    "message": self.history_db.message,
                    **self.history_db.details,
                },
            },
            "capabilities": {
                "generate_layout": self.can_generate,
                "preview_layout": self.can_preview,
                "generate_variations": self.can_generate,
                "rag_context": self.can_use_rag,
            },
        }


def check_kroki() -> ServiceStatus:
    """Check if Kroki rendering service is available."""
    try:
        from src.render import RenderClient

        client = RenderClient()
        if client.is_available():
            return ServiceStatus(
                available=True,
                message="Kroki service is running",
                details={"url": client.base_url},
            )
        else:
            return ServiceStatus(
                available=False,
                message="Kroki not responding. Run: python . docker up",
                details={"url": client.base_url},
            )
    except Exception as e:
        return ServiceStatus(
            available=False,
            message=f"Kroki check failed: {e}",
        )


def check_rag_index() -> ServiceStatus:
    """Check if RAG vector index is available."""
    try:
        from src.config import get_index_dir

        index_dir = get_index_dir()
        index_path = index_dir / "index.faiss"

        if not index_path.exists():
            return ServiceStatus(
                available=False,
                message="RAG index not found. Run: python . dev index build",
                details={"path": str(index_dir)},
            )

        # Check index size
        from src.vector import VectorStore

        store = VectorStore()
        store.load(index_dir)
        item_count = len(store)

        return ServiceStatus(
            available=True,
            message=f"RAG index loaded with {item_count} items",
            details={
                "path": str(index_dir),
                "item_count": item_count,
            },
        )
    except Exception as e:
        return ServiceStatus(
            available=False,
            message=f"RAG index check failed: {e}",
        )


def check_llm_providers() -> ServiceStatus:
    """Check which LLM providers are available."""
    try:
        from src.config import get_available_llm_providers

        providers = get_available_llm_providers()

        if not providers:
            return ServiceStatus(
                available=False,
                message="No LLM providers configured. Set API keys in .env",
                details={"available_providers": []},
            )

        return ServiceStatus(
            available=True,
            message=f"LLM providers available: {', '.join(providers)}",
            details={"available_providers": providers},
        )
    except Exception as e:
        return ServiceStatus(
            available=False,
            message=f"LLM provider check failed: {e}",
        )


def check_history_db() -> ServiceStatus:
    """Check if history database is accessible."""
    try:
        from src.history import get_history_manager

        manager = get_history_manager()
        stats = manager.get_stats()

        return ServiceStatus(
            available=True,
            message=f"History DB: {stats.artifact_count} artifacts",
            details={
                "artifact_count": stats.artifact_count,
                "session_count": stats.session_count,
                "size_mb": stats.total_size_mb,
            },
        )
    except Exception as e:
        return ServiceStatus(
            available=False,
            message=f"History DB check failed: {e}",
        )


def get_server_health() -> ServerHealth:
    """Get comprehensive server health status.

    Checks all dependencies and returns overall health assessment.

    Returns:
        ServerHealth with status of all services.
    """
    from .lib import get_server_version

    # Check all services
    kroki = check_kroki()
    rag_index = check_rag_index()
    llm_providers = check_llm_providers()
    history_db = check_history_db()

    # Determine overall status
    can_generate = llm_providers.available
    can_preview = kroki.available
    can_use_rag = rag_index.available

    if can_generate and can_preview:
        if can_use_rag:
            status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.DEGRADED  # Works but no RAG
    elif can_generate:
        status = HealthStatus.DEGRADED  # Can generate but not preview
    else:
        status = HealthStatus.UNHEALTHY  # Can't generate at all

    return ServerHealth(
        status=status,
        version=get_server_version(),
        checked_at=datetime.now(UTC),
        kroki=kroki,
        rag_index=rag_index,
        llm_providers=llm_providers,
        history_db=history_db,
        can_generate=can_generate,
        can_preview=can_preview,
        can_use_rag=can_use_rag,
    )


def log_startup_status() -> None:
    """Log server health status on startup.

    Called during server initialization to warn about missing dependencies.
    """
    health = get_server_health()

    logger.info(f"Server health: {health.status.value}")

    if not health.can_generate:
        logger.warning(
            f"LLM providers unavailable: {health.llm_providers.message}"
        )

    if not health.can_preview:
        logger.warning(
            f"Preview unavailable: {health.kroki.message}"
        )

    if not health.can_use_rag:
        logger.info(
            f"RAG context unavailable: {health.rag_index.message}"
        )

    if health.status == HealthStatus.UNHEALTHY:
        logger.error(
            "Server is UNHEALTHY - some tools will fail. "
            "Check configuration and services."
        )
    elif health.status == HealthStatus.DEGRADED:
        logger.warning(
            "Server is DEGRADED - some features unavailable. "
            "Generation will work but may lack context or previews."
        )


__all__ = [
    "HealthStatus",
    "ServiceStatus",
    "ServerHealth",
    "check_kroki",
    "check_rag_index",
    "check_llm_providers",
    "check_history_db",
    "get_server_health",
    "log_startup_status",
]
