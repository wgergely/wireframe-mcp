"""Docker container execution helpers.

Provides utilities for running CLI commands inside Docker containers,
with support for GPU passthrough and volume mounts.
"""

import subprocess


def build_docker_exec_command(
    command: list[str],
    container: str | None = None,
    image: str | None = None,
    gpu: bool = False,
    volumes: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    workdir: str = "/app",
) -> list[str]:
    """Build a docker exec or run command.

    Args:
        command: Command to run inside container.
        container: Container name for exec (mutually exclusive with image).
        image: Image name for run (mutually exclusive with container).
        gpu: Enable GPU passthrough (--gpus all).
        volumes: Volume mounts {host_path: container_path}.
        env: Environment variables.
        workdir: Working directory inside container.

    Returns:
        Complete docker command as list of strings.

    Raises:
        ValueError: If neither container nor image specified, or if both are.
    """
    if container and image:
        raise ValueError("Specify either container or image, not both")

    if container:
        # Use docker exec for running container
        docker_cmd = ["docker", "exec"]
        if env:
            for k, v in env.items():
                docker_cmd.extend(["-e", f"{k}={v}"])
        docker_cmd.append(container)
        docker_cmd.extend(command)
        return docker_cmd

    if image:
        # Use docker run for new container
        docker_cmd = ["docker", "run", "--rm"]

        if gpu:
            docker_cmd.extend(["--gpus", "all"])

        if workdir:
            docker_cmd.extend(["-w", workdir])

        if volumes:
            for host, cont in volumes.items():
                docker_cmd.extend(["-v", f"{host}:{cont}"])

        if env:
            for k, v in env.items():
                docker_cmd.extend(["-e", f"{k}={v}"])

        docker_cmd.append(image)
        docker_cmd.extend(command)
        return docker_cmd

    raise ValueError("Must specify either container or image")


def run_in_container(
    command: list[str],
    container: str | None = None,
    image: str | None = None,
    gpu: bool = False,
    volumes: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command inside a Docker container.

    Args:
        command: Command to run.
        container: Running container name (for exec).
        image: Image name (for run).
        gpu: Enable GPU.
        volumes: Volume mounts.
        env: Environment variables.
        capture_output: Capture stdout/stderr.

    Returns:
        Completed process with return code.
    """
    docker_cmd = build_docker_exec_command(
        command=command,
        container=container,
        image=image,
        gpu=gpu,
        volumes=volumes,
        env=env,
    )

    return subprocess.run(
        docker_cmd,
        capture_output=capture_output,
        text=True,
    )
