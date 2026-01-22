"""Environment detection implementation.

Detects GPU capabilities, platform constraints, and Docker availability
to help CLI commands make informed decisions about execution environment.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUCapabilities:
    """GPU/CUDA capability detection results.

    Attributes:
        faiss_available: Whether faiss is importable at all.
        faiss_gpu_available: Whether faiss-gpu with CUDA support is available.
        faiss_num_gpus: Number of GPUs detected by FAISS.
        torch_available: Whether PyTorch is importable.
        torch_cuda_available: Whether PyTorch CUDA is available.
        torch_cuda_devices: Number of CUDA devices detected by PyTorch.
    """

    faiss_available: bool = False
    faiss_gpu_available: bool = False
    faiss_num_gpus: int = 0
    torch_available: bool = False
    torch_cuda_available: bool = False
    torch_cuda_devices: int = 0

    @property
    def has_gpu(self) -> bool:
        """True if any GPU acceleration is available."""
        return self.faiss_gpu_available or self.torch_cuda_available

    @property
    def summary(self) -> str:
        """Human-readable summary of GPU capabilities."""
        parts = []
        if self.faiss_gpu_available:
            parts.append(f"FAISS-GPU ({self.faiss_num_gpus} GPU)")
        elif self.faiss_available:
            parts.append("FAISS-CPU only")
        else:
            parts.append("FAISS not installed")

        if self.torch_cuda_available:
            parts.append(f"PyTorch CUDA ({self.torch_cuda_devices} GPU)")
        elif self.torch_available:
            parts.append("PyTorch CPU only")
        else:
            parts.append("PyTorch not installed")

        return ", ".join(parts)


@dataclass(frozen=True)
class PlatformInfo:
    """Platform detection results.

    Attributes:
        platform: Raw sys.platform value.
        is_windows: True if running on Windows.
        is_linux: True if running on Linux.
        is_macos: True if running on macOS.
        gpu_support_limited: True if platform has limited GPU support.
    """

    platform: str
    is_windows: bool = False
    is_linux: bool = False
    is_macos: bool = False
    gpu_support_limited: bool = False

    @property
    def summary(self) -> str:
        """Human-readable platform summary."""
        name = "Windows" if self.is_windows else "Linux" if self.is_linux else "macOS" if self.is_macos else self.platform
        if self.gpu_support_limited:
            return f"{name} (GPU support limited)"
        return name


@dataclass(frozen=True)
class DockerInfo:
    """Docker availability detection results.

    Attributes:
        available: Whether docker CLI is available.
        version: Docker version string if available.
        nvidia_runtime: Whether nvidia container runtime is available.
        can_run_gpu: Whether Docker can run GPU workloads.
    """

    available: bool = False
    version: str | None = None
    nvidia_runtime: bool = False
    can_run_gpu: bool = False

    @property
    def summary(self) -> str:
        """Human-readable Docker summary."""
        if not self.available:
            return "Docker not available"
        gpu_note = " with GPU support" if self.can_run_gpu else " (no GPU runtime)"
        return f"Docker {self.version}{gpu_note}"


@dataclass
class EnvironmentReport:
    """Complete environment validation report.

    Attributes:
        gpu: GPU capability detection results.
        platform: Platform detection results.
        docker: Docker availability results.
        warnings: List of warning messages for the user.
        suggestions: List of suggested actions.
        can_run_gpu_commands: Whether GPU-dependent commands can run locally.
    """

    gpu: GPUCapabilities
    platform: PlatformInfo
    docker: DockerInfo
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    can_run_gpu_commands: bool = False

    def print_report(self) -> None:
        """Print formatted report to stdout."""
        print("=" * 60)
        print("Environment Report")
        print("=" * 60)
        print()
        print(f"Platform: {self.platform.summary}")
        print(f"GPU: {self.gpu.summary}")
        print(f"Docker: {self.docker.summary}")
        print()

        if self.warnings:
            print("Warnings:")
            for w in self.warnings:
                print(f"  ! {w}")
            print()

        if self.suggestions:
            print("Suggestions:")
            for s in self.suggestions:
                print(f"  > {s}")
            print()

        status = "Ready for GPU commands" if self.can_run_gpu_commands else "GPU commands may fail locally"
        print(f"Status: {status}")
        print()


def detect_gpu_capabilities() -> GPUCapabilities:
    """Detect GPU/CUDA capabilities.

    Checks both FAISS and PyTorch for GPU support. Safe to call
    even if libraries are not installed.

    Returns:
        GPUCapabilities with detection results.
    """
    faiss_available = False
    faiss_gpu_available = False
    faiss_num_gpus = 0
    torch_available = False
    torch_cuda_available = False
    torch_cuda_devices = 0

    # Check FAISS
    try:
        import faiss

        faiss_available = True
        # faiss-gpu has StandardGpuResources, faiss-cpu does not
        if hasattr(faiss, "StandardGpuResources"):
            if hasattr(faiss, "get_num_gpus"):
                faiss_num_gpus = faiss.get_num_gpus()
                faiss_gpu_available = faiss_num_gpus > 0
            else:
                # Has GPU support but can't detect count
                faiss_gpu_available = True
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"FAISS detection error: {e}")

    # Check PyTorch
    try:
        import torch

        torch_available = True
        torch_cuda_available = torch.cuda.is_available()
        if torch_cuda_available:
            torch_cuda_devices = torch.cuda.device_count()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"PyTorch detection error: {e}")

    return GPUCapabilities(
        faiss_available=faiss_available,
        faiss_gpu_available=faiss_gpu_available,
        faiss_num_gpus=faiss_num_gpus,
        torch_available=torch_available,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_devices=torch_cuda_devices,
    )


def detect_platform() -> PlatformInfo:
    """Detect current platform and its GPU support limitations.

    Windows has limited GPU support for FAISS (no faiss-gpu wheels).

    Returns:
        PlatformInfo with platform details.
    """
    platform = sys.platform
    is_windows = platform == "win32"
    is_linux = platform.startswith("linux")
    is_macos = platform == "darwin"

    # Windows: no faiss-gpu wheels available
    # macOS: limited CUDA support (Apple Silicon uses MPS)
    gpu_support_limited = is_windows or is_macos

    return PlatformInfo(
        platform=platform,
        is_windows=is_windows,
        is_linux=is_linux,
        is_macos=is_macos,
        gpu_support_limited=gpu_support_limited,
    )


def detect_docker() -> DockerInfo:
    """Detect Docker availability and GPU runtime support.

    Returns:
        DockerInfo with Docker detection results.
    """
    available = False
    version = None
    nvidia_runtime = False

    # Check docker CLI
    docker_path = shutil.which("docker")
    if docker_path:
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                available = True
                # Parse version from "Docker version X.Y.Z, build abc123"
                version_line = result.stdout.strip()
                if "version" in version_line.lower():
                    parts = version_line.split()
                    for i, p in enumerate(parts):
                        if p.lower() == "version" and i + 1 < len(parts):
                            version = parts[i + 1].rstrip(",")
                            break
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Docker version check failed: {e}")

    # Check nvidia container runtime
    if available:
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Runtimes}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "nvidia" in result.stdout.lower():
                nvidia_runtime = True
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Docker nvidia runtime check failed: {e}")

    can_run_gpu = available and nvidia_runtime

    return DockerInfo(
        available=available,
        version=version,
        nvidia_runtime=nvidia_runtime,
        can_run_gpu=can_run_gpu,
    )


def check_environment() -> EnvironmentReport:
    """Perform complete environment validation.

    Checks GPU capabilities, platform constraints, and Docker availability.
    Generates warnings and suggestions based on findings.

    Returns:
        EnvironmentReport with complete validation results.
    """
    gpu = detect_gpu_capabilities()
    platform = detect_platform()
    docker = detect_docker()

    warnings: list[str] = []
    suggestions: list[str] = []

    # Determine if GPU commands can run locally
    can_run_gpu = gpu.has_gpu

    # Platform-specific warnings
    if platform.is_windows:
        if not gpu.faiss_gpu_available:
            warnings.append(
                "Windows detected: FAISS-GPU wheels not available. "
                "GPU acceleration requires Docker."
            )
        if not gpu.torch_cuda_available and gpu.torch_available:
            warnings.append(
                "PyTorch installed without CUDA. For GPU support, reinstall with: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )

    # No GPU at all
    if not gpu.has_gpu:
        if platform.gpu_support_limited:
            warnings.append(
                f"No GPU acceleration available on {platform.summary}."
            )
            if docker.can_run_gpu:
                suggestions.append(
                    "Use --docker flag to run GPU commands in container."
                )
            elif docker.available:
                suggestions.append(
                    "Install NVIDIA Container Toolkit for Docker GPU support."
                )
            else:
                suggestions.append(
                    "Install Docker with NVIDIA runtime for GPU support."
                )
        else:
            warnings.append("No GPU detected. Commands will use CPU (slower).")

    # Docker suggestions
    if not docker.available:
        suggestions.append(
            "Install Docker for containerized GPU execution."
        )
    elif not docker.nvidia_runtime:
        suggestions.append(
            "Install NVIDIA Container Toolkit: "
            "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        )

    # FAISS not installed at all
    if not gpu.faiss_available:
        warnings.append(
            "FAISS not installed. Install with: pip install faiss-cpu"
        )

    return EnvironmentReport(
        gpu=gpu,
        platform=platform,
        docker=docker,
        warnings=warnings,
        suggestions=suggestions,
        can_run_gpu_commands=can_run_gpu,
    )


def require_gpu_or_warn(command_name: str) -> EnvironmentReport:
    """Check environment and warn if GPU not available.

    Call this at the start of GPU-dependent commands to provide
    clear feedback to users.

    Args:
        command_name: Name of the command being run (for error messages).

    Returns:
        EnvironmentReport for further decision making.
    """
    report = check_environment()

    if not report.can_run_gpu_commands:
        logger.warning(f"Command '{command_name}' requires GPU for best performance.")
        for w in report.warnings:
            logger.warning(f"  {w}")
        for s in report.suggestions:
            logger.info(f"  Suggestion: {s}")

    return report


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
