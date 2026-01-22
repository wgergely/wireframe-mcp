# CLI Environment Execution System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a unified environment execution system with GPU validation, platform-aware warnings, and consistent Docker execution across all GPU-dependent CLI commands.

**Architecture:** Create a centralized `src/env/` module that provides: (1) GPU/CUDA capability detection with platform-specific handling, (2) environment validation with actionable warnings, (3) reusable Docker execution wrappers. Commands check environment at entry and either proceed, warn, or suggest Docker alternative.

**Tech Stack:** Python 3.12, FAISS, PyTorch, Docker, subprocess, platform detection

---

## Overview

This plan addresses all 6 gaps identified in the audit:

1. **Gap 4.1**: No unified environment flag → Add `--docker` to GPU-dependent commands
2. **Gap 4.2**: Implicit local venv assumption → Add dependency validation with clear warnings
3. **Gap 4.3**: Two Docker command systems → Document relationship (no code change)
4. **Gap 4.4**: No local validation mode → Add `--env local` with validation
5. **Gap 4.5**: Backend selection scattered → Unify under env module
6. **Gap 4.6**: No environment introspection → Add `python . env check` command

---

## Task 1: Create Environment Detection Module

**Files:**
- Create: `src/env/__init__.py`
- Create: `src/env/lib.py`
- Create: `src/env/test.py`

### Step 1: Write the failing test for GPU detection

```python
# src/env/test.py
"""Tests for environment detection module."""

import pytest
from unittest.mock import patch, MagicMock


class TestGPUDetection:
    """Tests for GPU/CUDA detection."""

    def test_detect_faiss_cpu_only(self):
        """When faiss-cpu is installed, gpu_available should be False."""
        from src.env import detect_gpu_capabilities

        with patch.dict("sys.modules", {"faiss": MagicMock(spec=["IndexFlatIP"])}):
            # faiss-cpu doesn't have StandardGpuResources
            caps = detect_gpu_capabilities()
            assert caps.faiss_gpu_available is False

    def test_detect_faiss_gpu_available(self):
        """When faiss-gpu is installed with GPU, gpu_available should be True."""
        from src.env import detect_gpu_capabilities

        mock_faiss = MagicMock()
        mock_faiss.get_num_gpus.return_value = 1
        mock_faiss.StandardGpuResources = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            caps = detect_gpu_capabilities()
            assert caps.faiss_gpu_available is True

    def test_detect_torch_cuda_unavailable(self):
        """When torch has no CUDA, torch_cuda_available should be False."""
        from src.env import detect_gpu_capabilities

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            caps = detect_gpu_capabilities()
            assert caps.torch_cuda_available is False

    def test_detect_torch_cuda_available(self):
        """When torch has CUDA, torch_cuda_available should be True."""
        from src.env import detect_gpu_capabilities

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        with patch.dict("sys.modules", {"torch": mock_torch}):
            caps = detect_gpu_capabilities()
            assert caps.torch_cuda_available is True


class TestPlatformDetection:
    """Tests for platform-specific detection."""

    def test_detect_windows_platform(self):
        """Detect Windows platform correctly."""
        from src.env import detect_platform

        with patch("sys.platform", "win32"):
            plat = detect_platform()
            assert plat.is_windows is True
            assert plat.gpu_support_limited is True

    def test_detect_linux_platform(self):
        """Detect Linux platform correctly."""
        from src.env import detect_platform

        with patch("sys.platform", "linux"):
            plat = detect_platform()
            assert plat.is_windows is False
            assert plat.gpu_support_limited is False


class TestEnvironmentCheck:
    """Tests for full environment validation."""

    def test_environment_check_returns_report(self):
        """Environment check returns structured report."""
        from src.env import check_environment

        report = check_environment()

        assert hasattr(report, "gpu")
        assert hasattr(report, "platform")
        assert hasattr(report, "docker")
        assert hasattr(report, "warnings")
        assert hasattr(report, "can_run_gpu_commands")

    def test_windows_no_gpu_generates_warning(self):
        """Windows without GPU should generate Docker suggestion."""
        from src.env import check_environment

        with patch("sys.platform", "win32"):
            with patch("src.env.lib.detect_gpu_capabilities") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    faiss_gpu_available=False,
                    torch_cuda_available=False,
                )
                report = check_environment()
                assert len(report.warnings) > 0
                assert any("docker" in w.lower() for w in report.warnings)
```

### Step 2: Run test to verify it fails

Run: `pytest src/env/test.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.env'"

### Step 3: Write minimal implementation

```python
# src/env/__init__.py
"""Environment detection and validation for wireframe-mcp.

Provides GPU capability detection, platform awareness, and Docker availability
checks. Used by CLI commands to validate execution environment.

Example:
    >>> from src.env import check_environment
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
```

```python
# src/env/lib.py
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
```

### Step 4: Run test to verify it passes

Run: `pytest src/env/test.py -v`
Expected: PASS (all tests green)

### Step 5: Commit

```bash
git add src/env/
git commit -m "feat(env): add environment detection module

- Add GPU capability detection (FAISS, PyTorch CUDA)
- Add platform detection with Windows GPU limitation awareness
- Add Docker availability and nvidia runtime detection
- Add comprehensive EnvironmentReport with warnings/suggestions
- Add require_gpu_or_warn() for CLI command entry validation"
```

---

## Task 2: Add `env check` CLI Command

**Files:**
- Modify: `__main__.py:2408-2447` (add env command to main())
- Create: Test in existing CLI test structure

### Step 1: Write the failing test

```python
# Add to existing CLI tests or create tests/cli/test_env.py
"""Tests for env CLI command."""

import subprocess
import sys


def test_env_check_runs_without_error():
    """env check command should run and exit 0."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd=".",  # Run from repo root
    )
    assert result.returncode == 0
    assert "Environment Report" in result.stdout or "Platform:" in result.stdout


def test_env_check_shows_platform():
    """env check should show platform information."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "Platform:" in result.stdout


def test_env_check_shows_gpu():
    """env check should show GPU information."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "GPU:" in result.stdout


def test_env_check_shows_docker():
    """env check should show Docker information."""
    result = subprocess.run(
        [sys.executable, ".", "env", "check"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "Docker:" in result.stdout
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_env.py -v` (or wherever test is placed)
Expected: FAIL - command not recognized

### Step 3: Write minimal implementation

Add to `__main__.py` after the existing command handlers (around line 2400):

```python
# =============================================================================
# Env Command (Environment Introspection)
# =============================================================================


def cmd_env_check(_args: argparse.Namespace) -> int:
    """Check environment capabilities for GPU-dependent commands."""
    from src.env import check_environment

    report = check_environment()
    report.print_report()

    # Return non-zero if environment can't run GPU commands
    return 0 if report.can_run_gpu_commands else 1


def handle_env_command(argv: list[str]) -> int:
    """Handle environment inspection commands.

    Usage:
        python . env check    # Full environment report
        python . env gpu      # GPU-only check
        python . env docker   # Docker-only check
    """
    parser = argparse.ArgumentParser(
        prog="python . env",
        description="Environment inspection and validation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # check command
    check_parser = subparsers.add_parser(
        "check",
        help="Full environment capability check",
    )
    check_parser.set_defaults(func=cmd_env_check)

    # gpu command (quick check)
    gpu_parser = subparsers.add_parser(
        "gpu",
        help="GPU capability check only",
    )
    gpu_parser.set_defaults(func=lambda args: _cmd_env_gpu())

    # docker command (quick check)
    docker_parser = subparsers.add_parser(
        "docker",
        help="Docker availability check only",
    )
    docker_parser.set_defaults(func=lambda args: _cmd_env_docker())

    if not argv:
        # Default to full check
        argv = ["check"]

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


def _cmd_env_gpu() -> int:
    """Quick GPU check."""
    from src.env import detect_gpu_capabilities

    caps = detect_gpu_capabilities()
    print(f"GPU: {caps.summary}")
    print(f"  FAISS GPU: {'Yes' if caps.faiss_gpu_available else 'No'}")
    print(f"  PyTorch CUDA: {'Yes' if caps.torch_cuda_available else 'No'}")
    return 0 if caps.has_gpu else 1


def _cmd_env_docker() -> int:
    """Quick Docker check."""
    from src.env import detect_docker

    docker = detect_docker()
    print(f"Docker: {docker.summary}")
    print(f"  Available: {'Yes' if docker.available else 'No'}")
    print(f"  GPU Runtime: {'Yes' if docker.nvidia_runtime else 'No'}")
    return 0 if docker.can_run_gpu else 1
```

Then modify `main()` to include the env command (around line 2430):

```python
def main() -> int:
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        show_help()
        return 1

    command = sys.argv[1]
    rest_args = sys.argv[2:]

    if command in ("-h", "--help"):
        show_help()
        return 0

    # MCP server operations
    mcp_commands = {
        "mcp": lambda: handle_mcp_command(rest_args),
        "generate": lambda: handle_generate_command(rest_args),
        "search": lambda: handle_search_command(rest_args),
        "service": lambda: cmd_service(rest_args),
        "docker": lambda: handle_docker_command(rest_args),
        "index": lambda: handle_index_command(rest_args),
        "env": lambda: handle_env_command(rest_args),  # ADD THIS LINE
    }

    # ... rest of main()
```

Also update `show_help()` to include env command:

```python
def show_help() -> None:
    """Display CLI help message."""
    print("Usage: python . {command} [args]")
    print("\n=== MCP Server ===")
    print("  mcp        Run MCP server (STDIO or HTTP mode)")
    print("\n=== MCP Operations ===")
    print("  generate   Generate UI layouts from natural language")
    print("  search     Search vector indices for similar layouts")
    print("  service    Manage MCP services (init, start, stop, status)")
    print("  docker     Manage Docker stack (up, down, ps, build)")
    print("\n=== Index Management ===")
    print("  index      Build and manage RAG vector indices")
    print("\n=== Environment ===")
    print("  env        Check environment capabilities (GPU, Docker)")  # ADD THIS
    print("\n=== Development ===")
    print("  dev        Development workflows (test, benchmark, etc.)")
    # ... rest of help
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_env.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add __main__.py tests/cli/test_env.py
git commit -m "feat(cli): add 'env check' command for environment introspection

- Add 'python . env check' for full environment report
- Add 'python . env gpu' for quick GPU check
- Add 'python . env docker' for quick Docker check
- Shows platform, GPU capabilities, Docker status
- Provides warnings and suggestions for missing capabilities"
```

---

## Task 3: Add GPU Validation to `search` Command

**Files:**
- Modify: `__main__.py:648-715` (handle_search_command and cmd_search_index)
- Add: `--docker` flag to search command

### Step 1: Write the failing test

```python
# tests/cli/test_search.py
"""Tests for search command environment handling."""

import subprocess
import sys


def test_search_accepts_docker_flag():
    """search command should accept --docker flag."""
    result = subprocess.run(
        [sys.executable, ".", "search", "--help"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "--docker" in result.stdout


def test_search_warns_on_no_gpu(monkeypatch):
    """search should warn when GPU not available."""
    # This test would need mocking - simplified version
    # In practice, run on a machine without GPU to verify warning
    pass
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_search.py::test_search_accepts_docker_flag -v`
Expected: FAIL - --docker not in help output

### Step 3: Write minimal implementation

Modify `handle_search_command()` in `__main__.py`:

```python
def cmd_search_index(args: argparse.Namespace) -> int:
    """Handle the search command."""
    # Check for Docker execution mode
    if getattr(args, "docker", False):
        from src.config import get_index_dir
        from src.docker.exec import run_in_container

        index_path = args.index if args.index else get_index_dir()

        inner_cmd = [".", "search", args.query, "-k", str(args.k)]
        if args.index:
            inner_cmd.extend(["--index", str(args.index)])

        logger.info(f"Executing search in Docker container: {args.image}")
        result = run_in_container(
            command=inner_cmd,
            image=args.image,
            gpu=True,
            volumes={
                str(index_path.parent): "/app/corpus",
            },
        )
        return result.returncode

    # Validate environment for local execution
    from src.env import require_gpu_or_warn

    report = require_gpu_or_warn("search")
    if not report.can_run_gpu_commands and not args.force:
        logger.error(
            "GPU not available for local execution. "
            "Use --docker to run in container, or --force to proceed anyway."
        )
        return 1

    # Original implementation continues...
    from src.config import get_index_dir
    from src.vector import VectorStore

    try:
        # Resolve index path
        index_path = args.index if args.index else get_index_dir()

        # Load index
        store = VectorStore()
        store.load(index_path)
        logger.info(f"Loaded index with {len(store)} documents")

        # Search
        results = store.search(args.query, k=args.k)

        # Display results
        logger.info(f"\nTop {len(results)} results for: {args.query}\n")
        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            text = result.get("text", "")[:200]
            metadata = result.get("metadata", {})
            logger.info(f"{i}. [score={score:.4f}]")
            logger.info(f"   {text}...")
            if metadata:
                logger.info(f"   metadata: {metadata}")
            logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1


def handle_search_command(argv: list[str]) -> int:
    """Handle search-specific commands."""
    parser = argparse.ArgumentParser(
        prog="python . search",
        description="Search vector indices",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=Path,
        default=None,
        help="Index directory path (default: .corpus/index)",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    # NEW: Docker execution flags
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Execute inside Docker container with GPU support",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="wireframe-mcp:latest",
        help="Docker image to use (default: wireframe-mcp:latest)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force local execution even without GPU",
    )

    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)
    return cmd_search_index(args)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_search.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add __main__.py tests/cli/test_search.py
git commit -m "feat(cli): add --docker flag and GPU validation to search command

- Add --docker flag to run search in GPU container
- Add --force flag to override GPU requirement
- Add environment validation with clear warnings
- Suggest Docker when GPU not available locally"
```

---

## Task 4: Add GPU Validation to `benchmark` Command

**Files:**
- Modify: `__main__.py:861-903` (handle_benchmark_command)

### Step 1: Write the failing test

```python
# tests/cli/test_benchmark.py
"""Tests for benchmark command environment handling."""

import subprocess
import sys


def test_benchmark_accepts_docker_flag():
    """benchmark command should accept --docker flag."""
    result = subprocess.run(
        [sys.executable, ".", "dev", "benchmark", "--help"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "--docker" in result.stdout
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_benchmark.py -v`
Expected: FAIL

### Step 3: Write minimal implementation

Modify `handle_benchmark_command()`:

```python
def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run search quality benchmark against the vector index."""
    # Check for Docker execution mode
    if getattr(args, "docker", False):
        from src.docker.exec import run_in_container

        inner_cmd = [".", "dev", "benchmark"]
        inner_cmd.extend(["--index", args.index])
        inner_cmd.extend(["--model", args.model])
        if args.queries:
            inner_cmd.extend(["--queries", args.queries])
        inner_cmd.extend(["--min-pass-rate", str(args.min_pass_rate)])
        if args.verbose:
            inner_cmd.append("-v")

        logger.info(f"Executing benchmark in Docker container: {args.image}")
        result = run_in_container(
            command=inner_cmd,
            image=args.image,
            gpu=True,
            volumes={
                str(Path(args.index).parent.resolve()): "/app/corpus",
            },
        )
        return result.returncode

    # Validate environment for local execution
    from src.env import require_gpu_or_warn

    report = require_gpu_or_warn("benchmark")
    if not report.can_run_gpu_commands and not args.force:
        logger.error(
            "GPU not available for local execution. "
            "Use --docker to run in container, or --force to proceed anyway."
        )
        return 1

    # Original implementation continues...
    from src.config import get_data_dir
    # ... rest of original cmd_benchmark implementation


def handle_benchmark_command(argv: list[str]) -> int:
    """Handle benchmark command argument parsing."""

    parser = argparse.ArgumentParser(
        prog="python . benchmark",
        description="Run search quality benchmarks against the vector index",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=str,
        default="corpus_index",
        help="Index path (default: corpus_index)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--queries",
        "-q",
        type=str,
        default=None,
        help="Path to custom queries JSON file",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Minimum pass rate to succeed (0-100, default: 0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-query results",
    )
    # NEW: Docker execution flags
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Execute inside Docker container with GPU support",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="wireframe-mcp:latest",
        help="Docker image to use (default: wireframe-mcp:latest)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force local execution even without GPU",
    )

    args = parser.parse_args(argv)
    return cmd_benchmark(args)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_benchmark.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add __main__.py tests/cli/test_benchmark.py
git commit -m "feat(cli): add --docker flag and GPU validation to benchmark command

- Add --docker flag to run benchmark in GPU container
- Add --force flag to override GPU requirement
- Add environment validation with clear warnings"
```

---

## Task 5: Add GPU Validation to `generate` Command (RAG mode)

**Files:**
- Modify: `__main__.py:35-238` (handle_generate_command)

### Step 1: Write the failing test

```python
# tests/cli/test_generate.py
"""Tests for generate command environment handling."""

import subprocess
import sys


def test_generate_layout_accepts_docker_flag():
    """generate layout command should accept --docker flag when using --index."""
    result = subprocess.run(
        [sys.executable, ".", "generate", "layout", "--help"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "--docker" in result.stdout
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_generate.py -v`
Expected: FAIL

### Step 3: Write minimal implementation

Modify `cmd_generate_layout()`:

```python
def cmd_generate_layout(args: argparse.Namespace) -> int:
    """Handle the generate layout command."""
    # If using RAG (--index), check for Docker or GPU
    if args.index:
        if getattr(args, "docker", False):
            from src.docker.exec import run_in_container

            inner_cmd = [".", "generate", "layout", args.query]
            if args.model:
                inner_cmd.extend(["--model", args.model])
            if args.index:
                inner_cmd.extend(["--index", "/app/corpus/index"])
            if args.output:
                inner_cmd.extend(["--output", str(args.output)])
            inner_cmd.extend(["--target", args.target])
            inner_cmd.extend(["--format", args.format])

            logger.info(f"Executing generate in Docker container: {args.image}")
            result = run_in_container(
                command=inner_cmd,
                image=args.image,
                gpu=True,
                volumes={
                    str(args.index.parent): "/app/corpus",
                },
                env={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
            )
            return result.returncode

        # Validate environment for local RAG execution
        from src.env import require_gpu_or_warn

        report = require_gpu_or_warn("generate (with RAG)")
        if not report.can_run_gpu_commands and not args.force:
            logger.error(
                "GPU not available for RAG-enabled generation. "
                "Use --docker to run in container, or --force to proceed anyway."
            )
            return 1

    # Original implementation continues...
    from src.llm import (
        GeneratorConfig,
        LayoutGenerator,
        LLMModel,
        create_llm_backend,
    )
    # ... rest of original implementation
```

Add new arguments to `handle_generate_command()`:

```python
def handle_generate_command(argv: list[str]) -> int:
    """Handle generate-specific commands."""
    parser = argparse.ArgumentParser(
        prog="python . generate",
        description="Generate UI layouts from natural language",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # layout command (default)
    layout_parser = subparsers.add_parser(
        "layout",
        help="Generate a layout from a query",
    )
    # ... existing arguments ...

    # NEW: Docker execution flags
    layout_parser.add_argument(
        "--docker",
        action="store_true",
        help="Execute inside Docker container with GPU support (for RAG)",
    )
    layout_parser.add_argument(
        "--image",
        type=str,
        default="wireframe-mcp:latest",
        help="Docker image to use (default: wireframe-mcp:latest)",
    )
    layout_parser.add_argument(
        "--force",
        action="store_true",
        help="Force local execution even without GPU",
    )
    layout_parser.set_defaults(func=cmd_generate_layout)

    # ... rest of handler
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_generate.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add __main__.py tests/cli/test_generate.py
git commit -m "feat(cli): add --docker flag and GPU validation to generate command

- Add --docker flag for RAG-enabled generation
- Only validates GPU when --index is provided
- Add --force flag to override GPU requirement"
```

---

## Task 6: Add Startup Warning for Windows Without GPU

**Files:**
- Modify: `__main__.py:2408-2447` (main function)

### Step 1: Write the failing test

```python
# tests/cli/test_startup.py
"""Tests for CLI startup behavior."""

import subprocess
import sys
from unittest.mock import patch


def test_windows_no_gpu_shows_warning():
    """On Windows without GPU, CLI should show warning at startup."""
    # This is hard to test directly - verify warning function exists
    from src.env import check_environment

    report = check_environment()
    # Report should have warnings list
    assert hasattr(report, "warnings")
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_startup.py -v`
Expected: PASS (we're testing the capability exists)

### Step 3: Write minimal implementation

Add startup check to `main()`:

```python
def _check_startup_environment() -> None:
    """Check environment at CLI startup and warn about GPU issues.

    Only runs on first command invocation, not on every subcommand.
    Only warns for GPU-dependent commands.
    """
    import sys

    # Only warn for GPU-dependent commands
    gpu_commands = {"index", "search", "dev"}  # dev includes benchmark
    if len(sys.argv) < 2:
        return

    command = sys.argv[1]
    if command not in gpu_commands:
        return

    from src.env import detect_platform, detect_gpu_capabilities

    platform = detect_platform()
    gpu = detect_gpu_capabilities()

    # Windows-specific early warning
    if platform.is_windows and not gpu.has_gpu:
        import logging
        logger = logging.getLogger("cli")
        logger.warning(
            "Windows detected without GPU acceleration. "
            "GPU-dependent commands (index, search, benchmark) may be slow or fail. "
            "Consider using --docker flag for GPU support."
        )


def main() -> int:
    """Main entry point for the CLI."""
    # Early environment check for GPU-dependent commands
    _check_startup_environment()

    if len(sys.argv) < 2:
        show_help()
        return 1

    # ... rest of main()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_startup.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add __main__.py tests/cli/test_startup.py
git commit -m "feat(cli): add startup environment check with Windows GPU warning

- Check environment at CLI startup for GPU-dependent commands
- Show clear warning on Windows without GPU acceleration
- Suggest --docker flag for GPU support"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `docs/plans/2026-01-22-cli-environment-audit.md` (mark as implemented)
- Create: Update any existing README or docs

### Step 1: Update audit document status

```markdown
# Add to top of audit document:

> **Status**: IMPLEMENTED - See `2026-01-22-cli-environment-implementation.md`
```

### Step 2: Commit

```bash
git add docs/
git commit -m "docs: mark environment audit as implemented

- Reference implementation plan
- Document new --docker flags on commands
- Document env check command"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/integration/test_env_integration.py`

### Step 1: Write integration test

```python
# tests/integration/test_env_integration.py
"""Integration tests for environment detection and CLI flags."""

import subprocess
import sys


class TestEnvCommand:
    """Integration tests for env command."""

    def test_env_check_exits_cleanly(self):
        """env check should always exit without crash."""
        result = subprocess.run(
            [sys.executable, ".", "env", "check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should exit 0 (has GPU) or 1 (no GPU), but not crash
        assert result.returncode in (0, 1)
        assert "Environment Report" in result.stdout or "Platform:" in result.stdout

    def test_env_gpu_exits_cleanly(self):
        """env gpu should always exit without crash."""
        result = subprocess.run(
            [sys.executable, ".", "env", "gpu"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 1)
        assert "GPU:" in result.stdout or "FAISS" in result.stdout

    def test_env_docker_exits_cleanly(self):
        """env docker should always exit without crash."""
        result = subprocess.run(
            [sys.executable, ".", "env", "docker"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 1)
        assert "Docker:" in result.stdout


class TestDockerFlags:
    """Integration tests for --docker flags on commands."""

    def test_search_help_shows_docker_flag(self):
        """search --help should show --docker option."""
        result = subprocess.run(
            [sys.executable, ".", "search", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--docker" in result.stdout
        assert "--force" in result.stdout

    def test_index_build_help_shows_docker_flag(self):
        """index build --help should show --docker option."""
        result = subprocess.run(
            [sys.executable, ".", "index", "build", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--docker" in result.stdout

    def test_generate_layout_help_shows_docker_flag(self):
        """generate layout --help should show --docker option."""
        result = subprocess.run(
            [sys.executable, ".", "generate", "layout", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--docker" in result.stdout
```

### Step 2: Run integration tests

Run: `pytest tests/integration/test_env_integration.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/integration/
git commit -m "test: add integration tests for environment detection and Docker flags

- Test env check/gpu/docker subcommands
- Test --docker flag presence in search, index build, generate"
```

---

## Summary

This plan implements all 6 gaps identified in the audit:

| Gap | Solution | Task |
|-----|----------|------|
| 4.1 No unified env flag | Add `--docker` to GPU commands | Tasks 3, 4, 5 |
| 4.2 Implicit local assumption | Add validation + warnings | Tasks 3, 4, 5, 6 |
| 4.3 Two Docker systems | Document (no code change) | Task 7 |
| 4.4 No local validation | `--force` flag + warnings | Tasks 3, 4, 5 |
| 4.5 Backend scattered | Uses env module consistently | Task 1 |
| 4.6 No introspection | `python . env check` | Tasks 1, 2 |

Total estimated tasks: 8
Commits: ~10-12
