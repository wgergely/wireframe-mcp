"""CLI entry point for wireframe-mcp.

This module acts as the central entry point for the project's CLI tools.
It delegates commands to the appropriate submodules or runs specific tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.core import get_logger, setup_logging
from src.corpus.api import CorpusManager

logger = get_logger("cli")


def cmd_download(args: argparse.Namespace) -> int:
    """Handle the download command."""
    try:
        manager = CorpusManager(data_dir=args.data_dir)
        manager.fetch_dataset(args.provider, force=args.force)
        logger.info(f"Success! Dataset '{args.provider}' is ready.")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_datasets(_args: argparse.Namespace) -> int:
    """Handle the datasets command."""
    manager = CorpusManager()
    logger.info("Available Providers:")
    for name in manager.list_providers():
        logger.info(f"  {name}")
    return 0


def cmd_show_data_dir(args: argparse.Namespace) -> int:
    """Handle the data-dir command."""
    manager = CorpusManager(data_dir=args.data_dir)
    logger.info(f"Data directory: {manager.data_dir}")
    return 0


def handle_corpus_command(argv: list[str]) -> int:
    """Handle corpus-specific commands."""
    parser = argparse.ArgumentParser(
        prog="python . corpus",
        description="Corpus data management for wireframe-mcp",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download a corpus dataset",
    )
    download_parser.add_argument(
        "provider",
        type=str,
        help="Provider name (e.g. rico_semantic)",
    )
    download_parser.add_argument(
        "--data-dir",
        "-o",
        type=Path,
        default=None,
        help="Target data directory (default: ./data)",
    )
    download_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if data exists",
    )
    download_parser.set_defaults(func=cmd_download)

    # datasets command
    list_parser = subparsers.add_parser(
        "datasets",
        help="List available datasets",
    )
    list_parser.set_defaults(func=cmd_datasets)

    # data-dir command
    dir_parser = subparsers.add_parser(
        "data-dir",
        help="Show the data directory path",
    )
    dir_parser.add_argument(
        "--data-dir",
        "-o",
        type=Path,
        default=None,
        help="Override data directory",
    )
    dir_parser.set_defaults(func=cmd_show_data_dir)

    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


def cmd_test(extra_args: list[str]) -> int:
    """Run pytest with provided arguments."""
    cmd = [sys.executable, "-m", "pytest", *extra_args]
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def cmd_build(extra_args: list[str]) -> int:
    """Run docker compose commands."""
    compose_file = Path("docker/docker-compose.yml")
    if not compose_file.exists():
        logger.error(f"Error: {compose_file} not found.")
        return 1

    action = extra_args if extra_args else ["build"]
    final_cmd = ["docker", "compose", "-f", str(compose_file), *action]

    logger.info(f"Running: {' '.join(final_cmd)}")
    try:
        return subprocess.call(final_cmd)
    except KeyboardInterrupt:
        return 130


def show_help() -> None:
    """Display CLI help message."""
    print("Usage: python . {corpus|test|build} [args]")
    print("\nCommands:")
    print("  corpus   Manage corpus data (download, list)")
    print("  test     Run tests (wrapper for pytest)")
    print("  build    Manage docker containers (wrapper for docker compose)")
    print("\nExamples:")
    print("  python . corpus download rico_semantic")
    print("  python . test -v")
    print("  python . build up -d")


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

    commands = {
        "corpus": lambda: handle_corpus_command(rest_args),
        "test": lambda: cmd_test(rest_args),
        "build": lambda: cmd_build(rest_args),
    }

    if command in commands:
        setup_logging()
        return commands[command]()

    logger.error(f"Unknown command: {command}")
    show_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
