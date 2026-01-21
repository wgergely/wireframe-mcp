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

        # Check index size by reading metadata directly
        # (Avoids initializing embedding backend which may require API keys)
        import json

        store_meta_path = index_dir / "index.store.json"
        if store_meta_path.exists():
            with open(store_meta_path) as f:
                store_data = json.load(f)
                item_count = len(store_data.get("metadata", {}))
        else:
            # Fallback to FAISS metadata
            faiss_meta_path = index_dir / "index.meta.json"
            if faiss_meta_path.exists():
                with open(faiss_meta_path) as f:
                    faiss_data = json.load(f)
                    item_count = faiss_data.get("size", 0)
            else:
                item_count = 0

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


def format_startup_banner(health: ServerHealth) -> str:
    """Format a startup status banner for logging.

    Args:
        health: Server health status.

    Returns:
        Formatted multi-line banner string.
    """
    # Status indicators
    status_icon = {
        HealthStatus.HEALTHY: "[OK]",
        HealthStatus.DEGRADED: "[!!]",
        HealthStatus.UNHEALTHY: "[XX]",
    }

    def svc_icon(available: bool) -> str:
        return "[OK]" if available else "[--]"

    # Build service status lines
    llm_icon = svc_icon(health.llm_providers.available)
    kroki_icon = svc_icon(health.kroki.available)
    rag_icon = svc_icon(health.rag_index.available)
    hist_icon = svc_icon(health.history_db.available)

    # Capability status
    gen_status = "Yes" if health.can_generate else "No - needs LLM"
    prev_status = "Yes" if health.can_preview else "No - needs Kroki"
    rag_status = "Yes" if health.can_use_rag else "No - needs index"

    lines = [
        "",
        "=" * 60,
        f"  Wireframe MCP Server v{health.version}",
        "=" * 60,
        f"  Status: {status_icon[health.status]} {health.status.value.upper()}",
        "",
        "  Services:",
        f"    {llm_icon} LLM Providers: {health.llm_providers.message}",
        f"    {kroki_icon} Kroki Render:  {health.kroki.message}",
        f"    {rag_icon} RAG Index:     {health.rag_index.message}",
        f"    {hist_icon} History DB:    {health.history_db.message}",
        "",
        "  Capabilities:",
        f"    generate_layout:    {gen_status}",
        f"    preview_layout:     {prev_status}",
        f"    rag_context:        {rag_status}",
    ]

    # Add action items if not healthy
    if health.status != HealthStatus.HEALTHY:
        lines.append("")
        lines.append("  Action Required:")
        if not health.can_generate:
            lines.append("    - Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")
        if not health.can_preview:
            lines.append("    - Run: python . docker up")
        if not health.can_use_rag:
            lines.append("    - Run: python . dev index build")

    lines.extend(
        [
            "",
            "=" * 60,
            "",
        ]
    )

    return "\n".join(lines)


def log_startup_status() -> None:
    """Log server health status on startup.

    Called during server initialization to warn about missing dependencies.
    Outputs a formatted banner showing service status and capabilities.
    """
    health = get_server_health()

    # Log the formatted banner
    banner = format_startup_banner(health)
    for line in banner.split("\n"):
        if line.strip():
            logger.info(line)

    # Log summary message at appropriate level
    if health.status == HealthStatus.UNHEALTHY:
        logger.error(
            "Server is UNHEALTHY - generate_layout will fail. "
            "Configure LLM provider API keys to proceed."
        )
    elif health.status == HealthStatus.DEGRADED:
        logger.warning(
            "Server is DEGRADED - some features unavailable. "
            "Generation works but may lack preview or context."
        )
    else:
        logger.info("Server is ready - all features available.")


__all__ = [
    "HealthStatus",
    "ServiceStatus",
    "ServerHealth",
    "check_kroki",
    "check_rag_index",
    "check_llm_providers",
    "check_history_db",
    "get_server_health",
    "format_startup_banner",
    "log_startup_status",
]
