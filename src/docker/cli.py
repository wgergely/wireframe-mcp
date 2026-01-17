"""Docker management CLI for wireframe-mcp.

This module provides the `python . docker` command group for managing the
Docker stack. It wraps `docker compose` to ensure consistent configuration,
file selection, and environment variable handling.
"""

import argparse
import os
import subprocess
from pathlib import Path

from docker.lib import (
    PROJECT_NAME,
    get_compose_files,
    list_modes,
)
from src.core import get_logger

logger = get_logger("docker")


def _run_compose(
    files: list[Path],
    command: list[str],
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    """Run a docker compose command.

    Args:
        files: List of compose files to include.
        command: Command arguments (e.g. ["up", "-d"]).
        dry_run: Check files but don't execute.
        env: Environment variables to pass.

    Returns:
        Subprocess exit code.
    """
    args = ["docker", "compose", "-p", PROJECT_NAME]
    for file in files:
        args.extend(["-f", str(file)])

    args.extend(command)

    cmd_str = " ".join(args)
    logger.info(f"Running: {cmd_str}")

    if dry_run:
        return 0

    try:
        # Pass current environment + overrides
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        return subprocess.call(args, env=run_env)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed to run docker compose: {e}")
        return 1


def cmd_up(args: argparse.Namespace) -> int:
    """Handle `docker up`."""
    try:
        files = get_compose_files(
            mode=args.mode,
            include_kroki=not args.no_kroki,
            include_server=not args.no_server,
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    cmd_args = ["up"]
    if args.detach:
        cmd_args.append("-d")
    if args.build:
        cmd_args.append("--build")

    # Pass arguments to specific services if provided
    # Note: docker compose up [SERVICE...]
    if args.services:
        cmd_args.extend(args.services)

    return _run_compose(files, cmd_args, dry_run=args.dry_run)


def cmd_down(args: argparse.Namespace) -> int:
    """Handle `docker down`."""
    # Down needs to know the files to tear down properly if project name isn't enough,
    # but usually -p PROJECT_NAME is enough for modern compose.
    # However, passing files is safer to ensure all volumes/networks are seen.
    try:
        # Load all potential files to be safe, or just base + server + kroki
        files = get_compose_files(mode=args.mode)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    cmd_args = ["down"]
    if args.volumes:
        cmd_args.append("-v")
    if args.remove_orphans:
        cmd_args.append("--remove-orphans")

    return _run_compose(files, cmd_args, dry_run=args.dry_run)


def cmd_ps(args: argparse.Namespace) -> int:
    """Handle `docker ps`."""
    try:
        files = get_compose_files(mode=args.mode)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    return _run_compose(files, ["ps"], dry_run=args.dry_run)


def cmd_build(args: argparse.Namespace) -> int:
    """Handle `docker build`."""
    try:
        files = get_compose_files(
            mode=args.mode,
            include_kroki=not args.no_kroki,
            include_server=not args.no_server,
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    cmd_args = ["build"]
    if args.no_cache:
        cmd_args.append("--no-cache")
    if args.services:
        cmd_args.extend(args.services)

    return _run_compose(files, cmd_args, dry_run=args.dry_run)


def cmd_proxy(args: argparse.Namespace) -> int:
    """Handle generic proxy commands (exec, run, logs, etc)."""
    try:
        files = get_compose_files(
            mode=args.mode,
            include_kroki=not getattr(args, "no_kroki", False),
            include_server=not getattr(args, "no_server", False),
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Construct the command: [subcommand] + [args...]
    # e.g. "exec" + ["-it", "server", "bash"]
    cmd_args = [args.command] + args.rest
    return _run_compose(files, cmd_args, dry_run=args.dry_run)


def handle_docker_command(argv: list[str]) -> int:
    """Handle docker-specific commands."""
    parser = argparse.ArgumentParser(
        prog="python . docker",
        description="Manage the Docker stack",
    )

    # Global arguments
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="dev",
        choices=list_modes(),
        help="Deployment mode (maps to compose.{mode}.yml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command but do not execute",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # up
    up_parser = subparsers.add_parser("up", help="Start services")
    up_parser.add_argument(
        "-d", "--detach", action="store_true", help="Run in background"
    )
    up_parser.add_argument(
        "--build", action="store_true", help="Build images before starting"
    )
    up_parser.add_argument(
        "--no-kroki", action="store_true", help="Exclude Kroki services"
    )
    up_parser.add_argument(
        "--no-server", action="store_true", help="Exclude MCP server"
    )
    up_parser.add_argument("services", nargs="*", help="Specific services to start")
    up_parser.set_defaults(func=cmd_up)

    # down
    down_parser = subparsers.add_parser("down", help="Stop and remove services")
    down_parser.add_argument(
        "-v", "--volumes", action="store_true", help="Remove volumes"
    )
    down_parser.add_argument(
        "--remove-orphans",
        action="store_true",
        help="Remove containers for services not defined",
    )
    down_parser.set_defaults(func=cmd_down)

    # ps
    ps_parser = subparsers.add_parser("ps", help="List running services")
    ps_parser.set_defaults(func=cmd_ps)

    # build
    build_parser = subparsers.add_parser("build", help="Build or rebuild services")
    build_parser.add_argument(
        "--no-cache", action="store_true", help="Do not use cache"
    )
    build_parser.add_argument(
        "--no-kroki", action="store_true", help="Exclude Kroki services"
    )
    build_parser.add_argument(
        "--no-server", action="store_true", help="Exclude MCP server"
    )
    build_parser.add_argument("services", nargs="*", help="Specific services to build")
    build_parser.set_defaults(func=cmd_build)

    # Proxy commands (exec, run, logs, etc.)
    # These capture all remaining arguments to pass to docker compose
    proxy_cmds = ["exec", "run", "logs", "pull", "push", "stop", "start", "restart"]
    for cmd in proxy_cmds:
        p = subparsers.add_parser(cmd, help=f"Proxy for `docker compose {cmd}`")
        p.add_argument(
            "rest", nargs=argparse.REMAINDER, help="Arguments passed to docker compose"
        )
        p.set_defaults(func=cmd_proxy)

    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)
