"""Tests for Docker container execution."""

from unittest.mock import patch

import pytest

from src.docker.exec import build_docker_exec_command, run_in_container


class TestBuildDockerExecCommand:
    """Tests for build_docker_exec_command function."""

    def test_build_exec_command_basic(self):
        """Test basic command building with container name."""
        cmd = build_docker_exec_command(
            command=["python", ".", "index", "build", "--all"],
            container="wfmcp-server",
        )
        assert cmd == [
            "docker",
            "exec",
            "wfmcp-server",
            "python",
            ".",
            "index",
            "build",
            "--all",
        ]

    def test_build_exec_command_with_env(self):
        """Test exec command with environment variables."""
        cmd = build_docker_exec_command(
            command=["python", ".", "index", "build"],
            container="wfmcp-server",
            env={"EMBEDDING_BACKEND": "local", "DEBUG": "1"},
        )
        assert "docker" in cmd
        assert "exec" in cmd
        assert "-e" in cmd
        assert "EMBEDDING_BACKEND=local" in cmd
        assert "DEBUG=1" in cmd
        assert "wfmcp-server" in cmd

    def test_build_run_command_with_gpu(self):
        """Test GPU-enabled run command."""
        cmd = build_docker_exec_command(
            command=["python", ".", "index", "build"],
            container=None,
            image="wireframe-mcp:latest",
            gpu=True,
        )
        assert "--gpus" in cmd
        assert "all" in cmd
        assert "wireframe-mcp:latest" in cmd

    def test_build_run_command_with_volumes(self):
        """Test run command with volume mounts."""
        cmd = build_docker_exec_command(
            command=["python", ".", "index", "build"],
            image="wireframe-mcp:latest",
            volumes={"/host/data": "/app/data", "/host/models": "/app/models"},
        )
        assert "-v" in cmd
        assert "/host/data:/app/data" in cmd
        assert "/host/models:/app/models" in cmd

    def test_build_run_command_with_workdir(self):
        """Test run command with custom working directory."""
        cmd = build_docker_exec_command(
            command=["python", ".", "index", "build"],
            image="wireframe-mcp:latest",
            workdir="/custom/path",
        )
        assert "-w" in cmd
        assert "/custom/path" in cmd

    def test_build_run_command_with_env(self):
        """Test run command with environment variables."""
        cmd = build_docker_exec_command(
            command=["python", ".", "index", "build"],
            image="wireframe-mcp:latest",
            env={"VOYAGE_API_KEY": "test-key"},
        )
        assert "-e" in cmd
        assert "VOYAGE_API_KEY=test-key" in cmd

    def test_error_neither_container_nor_image(self):
        """Test error when neither container nor image specified."""
        with pytest.raises(ValueError, match="Must specify either container or image"):
            build_docker_exec_command(command=["echo", "hello"])

    def test_error_both_container_and_image(self):
        """Test error when both container and image specified."""
        with pytest.raises(
            ValueError, match="Specify either container or image, not both"
        ):
            build_docker_exec_command(
                command=["echo", "hello"],
                container="wfmcp-server",
                image="wireframe-mcp:latest",
            )

    def test_run_command_includes_rm_flag(self):
        """Test that run commands include --rm for cleanup."""
        cmd = build_docker_exec_command(
            command=["python", ".", "index", "build"],
            image="wireframe-mcp:latest",
        )
        assert "--rm" in cmd


class TestRunInContainer:
    """Tests for run_in_container function."""

    @patch("src.docker.exec.subprocess.run")
    def test_run_in_container_with_image(self, mock_run):
        """Test run_in_container builds correct command and calls subprocess."""
        mock_run.return_value.returncode = 0

        run_in_container(
            command=["python", ".", "index", "build"],
            image="wireframe-mcp:latest",
            capture_output=True,
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]

        assert docker_cmd[0:3] == ["docker", "run", "--rm"]
        assert "-w" in docker_cmd
        assert "/app" in docker_cmd
        assert "wireframe-mcp:latest" in docker_cmd
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True

    @patch("src.docker.exec.subprocess.run")
    def test_run_in_container_with_custom_workdir(self, mock_run):
        """Test run_in_container passes custom workdir to docker command."""
        mock_run.return_value.returncode = 0

        run_in_container(
            command=["echo", "hello"],
            image="wireframe-mcp:latest",
            workdir="/custom/workdir",
        )

        mock_run.assert_called_once()
        docker_cmd = mock_run.call_args[0][0]

        assert "-w" in docker_cmd
        assert "/custom/workdir" in docker_cmd

    @patch("src.docker.exec.subprocess.run")
    def test_run_in_container_with_container(self, mock_run):
        """Test run_in_container with existing container (exec mode)."""
        mock_run.return_value.returncode = 0

        run_in_container(
            command=["echo", "hello"],
            container="wfmcp-server",
        )

        mock_run.assert_called_once()
        docker_cmd = mock_run.call_args[0][0]

        assert docker_cmd[0:2] == ["docker", "exec"]
        assert "wfmcp-server" in docker_cmd
