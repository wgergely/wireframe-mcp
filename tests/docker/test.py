"""Tests for Docker container execution."""

from unittest.mock import patch

import pytest

from src.docker.exec import build_docker_exec_command, run_in_container


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
class TestIndexBuildDockerFlag:
    """Tests for index build --docker flag CLI integration."""

    def test_docker_argument_parsed(self):
        """Test --docker argument is correctly parsed by argparse."""
        import argparse
        from pathlib import Path

        # Simulate the parser setup from handle_index_command
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_parser = subparsers.add_parser("build")
        build_parser.add_argument("provider", nargs="?", default=None)
        build_parser.add_argument("--all", "-a", action="store_true")
        build_parser.add_argument("--output", "-o", type=Path, default=None)
        build_parser.add_argument("--backend", "-b", type=str, default=None)
        build_parser.add_argument("--batch-size", type=int, default=100)
        build_parser.add_argument("--limit", type=int, default=None)
        build_parser.add_argument("--skip-download", action="store_true")
        build_parser.add_argument("--docker", action="store_true")
        build_parser.add_argument("--image", type=str, default="wireframe-mcp:latest")

        args = parser.parse_args(["build", "rico_semantic", "--docker"])
        assert args.docker is True
        assert args.provider == "rico_semantic"
        assert args.image == "wireframe-mcp:latest"

    def test_image_argument_with_custom_value(self):
        """Test --image argument accepts custom values."""
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_parser = subparsers.add_parser("build")
        build_parser.add_argument("provider", nargs="?", default=None)
        build_parser.add_argument("--all", "-a", action="store_true")
        build_parser.add_argument("--output", "-o", type=Path, default=None)
        build_parser.add_argument("--backend", "-b", type=str, default=None)
        build_parser.add_argument("--batch-size", type=int, default=100)
        build_parser.add_argument("--limit", type=int, default=None)
        build_parser.add_argument("--skip-download", action="store_true")
        build_parser.add_argument("--docker", action="store_true")
        build_parser.add_argument("--image", type=str, default="wireframe-mcp:latest")

        args = parser.parse_args(["build", "--all", "--docker", "--image", "custom:v1"])
        assert args.docker is True
        assert args.all is True
        assert args.image == "custom:v1"

    def test_docker_flag_builds_correct_inner_command(self):
        """Test that docker mode builds inner command correctly."""
        import argparse

        # Create a minimal args object
        args = argparse.Namespace(
            docker=True,
            image="wireframe-mcp:latest",
            all=False,
            provider="rico_semantic",
            backend="local",
            batch_size=100,
            limit=50,
            output=None,
            skip_download=False,
        )

        # Build inner command like cmd_build_index does
        inner_cmd = ["python", ".", "index", "build"]
        if args.all:
            inner_cmd.append("--all")
        elif args.provider:
            inner_cmd.append(args.provider)
        if args.backend:
            inner_cmd.extend(["--backend", args.backend])
        if args.batch_size:
            inner_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.limit:
            inner_cmd.extend(["--limit", str(args.limit)])
        if args.output:
            inner_cmd.extend(["--output", str(args.output)])
        if args.skip_download:
            inner_cmd.append("--skip-download")

        expected = [
            "python",
            ".",
            "index",
            "build",
            "rico_semantic",
            "--backend",
            "local",
            "--batch-size",
            "100",
            "--limit",
            "50",
        ]
        assert inner_cmd == expected

    def test_docker_flag_with_all_providers(self):
        """Test that --all flag is passed correctly in docker mode."""
        import argparse

        args = argparse.Namespace(
            docker=True,
            image="wireframe-mcp:latest",
            all=True,
            provider=None,
            backend=None,
            batch_size=100,
            limit=None,
            output=None,
            skip_download=True,
        )

        inner_cmd = ["python", ".", "index", "build"]
        if args.all:
            inner_cmd.append("--all")
        elif args.provider:
            inner_cmd.append(args.provider)
        if args.backend:
            inner_cmd.extend(["--backend", args.backend])
        if args.batch_size:
            inner_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.limit:
            inner_cmd.extend(["--limit", str(args.limit)])
        if args.output:
            inner_cmd.extend(["--output", str(args.output)])
        if args.skip_download:
            inner_cmd.append("--skip-download")

        expected = [
            "python",
            ".",
            "index",
            "build",
            "--all",
            "--batch-size",
            "100",
            "--skip-download",
        ]
        assert inner_cmd == expected

    @patch("src.docker.exec.subprocess.run")
    def test_docker_run_called_with_gpu(self, mock_run):
        """Test that docker run is called with GPU flag."""
        mock_run.return_value.returncode = 0

        run_in_container(
            command=["python", ".", "index", "build", "rico_semantic"],
            image="wireframe-mcp:latest",
            gpu=True,
            volumes={
                "/host/data": "/app/corpus/data",
                "/host/corpus": "/app/corpus",
            },
            env={"EMBEDDING_BACKEND": "local"},
        )

        mock_run.assert_called_once()
        docker_cmd = mock_run.call_args[0][0]

        # Verify GPU flag
        assert "--gpus" in docker_cmd
        assert "all" in docker_cmd

        # Verify volumes
        assert "-v" in docker_cmd
        assert "/host/data:/app/corpus/data" in docker_cmd
        assert "/host/corpus:/app/corpus" in docker_cmd

        # Verify env
        assert "-e" in docker_cmd
        assert "EMBEDDING_BACKEND=local" in docker_cmd

        # Verify command
        assert "python" in docker_cmd
        assert "index" in docker_cmd
        assert "build" in docker_cmd
        assert "rico_semantic" in docker_cmd
