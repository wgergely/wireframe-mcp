"""Tests for environment detection module."""

import pytest
from unittest.mock import patch, MagicMock


class TestGPUDetection:
    """Tests for GPU/CUDA detection."""

    def test_detect_faiss_cpu_only(self):
        """When faiss-cpu is installed, gpu_available should be False."""
        from src.environment import detect_gpu_capabilities

        with patch.dict("sys.modules", {"faiss": MagicMock(spec=["IndexFlatIP"])}):
            # faiss-cpu doesn't have StandardGpuResources
            caps = detect_gpu_capabilities()
            assert caps.faiss_gpu_available is False

    def test_detect_faiss_gpu_available(self):
        """When faiss-gpu is installed with GPU, gpu_available should be True."""
        from src.environment import detect_gpu_capabilities

        mock_faiss = MagicMock()
        mock_faiss.get_num_gpus.return_value = 1
        mock_faiss.StandardGpuResources = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            caps = detect_gpu_capabilities()
            assert caps.faiss_gpu_available is True

    def test_detect_torch_cuda_unavailable(self):
        """When torch has no CUDA, torch_cuda_available should be False."""
        from src.environment import detect_gpu_capabilities

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            caps = detect_gpu_capabilities()
            assert caps.torch_cuda_available is False

    def test_detect_torch_cuda_available(self):
        """When torch has CUDA, torch_cuda_available should be True."""
        from src.environment import detect_gpu_capabilities

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
        from src.environment import detect_platform

        with patch("src.environment.lib.sys.platform", "win32"):
            plat = detect_platform()
            assert plat.is_windows is True
            assert plat.gpu_support_limited is True

    def test_detect_linux_platform(self):
        """Detect Linux platform correctly."""
        from src.environment import detect_platform

        with patch("src.environment.lib.sys.platform", "linux"):
            plat = detect_platform()
            assert plat.is_windows is False
            assert plat.gpu_support_limited is False


class TestEnvironmentCheck:
    """Tests for full environment validation."""

    def test_environment_check_returns_report(self):
        """Environment check returns structured report."""
        from src.environment import check_environment

        report = check_environment()

        assert hasattr(report, "gpu")
        assert hasattr(report, "platform")
        assert hasattr(report, "docker")
        assert hasattr(report, "warnings")
        assert hasattr(report, "can_run_gpu_commands")

    def test_windows_no_gpu_generates_warning(self):
        """Windows without GPU should generate Docker suggestion."""
        from src.environment.lib import GPUCapabilities, check_environment

        with patch("src.environment.lib.sys.platform", "win32"):
            with patch("src.environment.lib.detect_gpu_capabilities") as mock_gpu:
                mock_gpu.return_value = GPUCapabilities(
                    faiss_available=True,
                    faiss_gpu_available=False,
                    torch_available=True,
                    torch_cuda_available=False,
                )
                report = check_environment()
                assert len(report.warnings) > 0
                assert any("docker" in w.lower() for w in report.warnings + report.suggestions)
