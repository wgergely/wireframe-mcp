"""CLI entry point for wireframe-mcp.

This module acts as the central entry point for the project's CLI tools.
It delegates commands to the appropriate submodules or runs specific tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.corpus.api import CorpusManager


def cmd_download(args: argparse.Namespace) -> int:
    """Handle the download command."""
    try:
        manager = CorpusManager(data_dir=args.data_dir)
        provider_name = args.provider

        manager.fetch_dataset(provider_name, force=args.force)

        # Let's just say "Success".
        print(f"\nSuccess! Dataset '{provider_name}' is ready.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_datasets(args: argparse.Namespace) -> int:
    """Handle the datasets command."""
    manager = CorpusManager()
    providers = manager.list_providers()
    print("Available Providers:\n")
    for name in providers:
        print(f"  {name}")
    return 0


def cmd_show_data_dir(args: argparse.Namespace) -> int:
    """Handle the data-dir command."""
    manager = CorpusManager(data_dir=args.data_dir)
    print(f"Data directory: {manager.data_dir}")
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


def cmd_test(extra_args):
    """Run pytest with provided arguments."""
    cmd = [sys.executable, "-m", "pytest"] + extra_args
    print(f"Running: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def cmd_build(extra_args):
    """Run docker compose commands."""
    compose_file = Path("src/docker/docker-compose.yml")
    if not compose_file.exists():
        print(f"Error: {compose_file} not found.")
        return 1

    base_cmd = ["docker", "compose", "-f", str(compose_file)]

    if not extra_args:
        # Default to build if no args provided
        final_cmd = base_cmd + ["build"]
    else:
        final_cmd = base_cmd + extra_args

    print(f"Running: {' '.join(final_cmd)}")
    try:
        return subprocess.call(final_cmd)
    except KeyboardInterrupt:
        return 130


def show_help():
    print("Usage: python . {corpus|test|build} [args]")
    print("\nCommands:")
    print("  corpus   Manage corpus data (download, list)")
    print("  test     Run tests (wrapper for pytest)")
    print("  build    Manage docker containers (wrapper for docker compose)")
    print("\nExamples:")
    print("  python . corpus download rico_semantic")
    print("  python . test -v")
    print("  python . build up -d")


def main():
    if len(sys.argv) < 2:
        show_help()
        return 1

    command = sys.argv[1]
    rest_args = sys.argv[2:]

    if command in ("-h", "--help"):
        show_help()
        return 0

    if command == "corpus":
        return handle_corpus_command(rest_args)

    elif command == "test":
        return cmd_test(rest_args)

    elif command == "build":
        return cmd_build(rest_args)

    else:
        print(f"Unknown command: {command}")
        show_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
