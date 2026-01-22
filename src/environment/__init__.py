"""Environment detection and validation for wireframe-mcp.

Provides GPU capability detection, platform awareness, and Docker availability
checks. Used by CLI commands to validate execution environment.

Example:
    >>> from src.environment import check_environment
    >>> report = check_environment()
    >>> if not report.can_run_gpu_commands:
    ...     print("GPU not available, consider using --docker")
"""

from .lib import (
    GPUCapabilities,
    PlatformInfo,
    DockerInfo,
    EnvironmentReport,
    detect_gpu_capabilities,
    detect_platform,
    detect_docker,
    check_environment,
    require_gpu_or_warn,
)

__all__ = [
    "GPUCapabilities",
    "PlatformInfo",
    "DockerInfo",
    "EnvironmentReport",
    "detect_gpu_capabilities",
    "detect_platform",
    "detect_docker",
    "check_environment",
    "require_gpu_or_warn",
]
