"""CLI entry point for corpus data management.

This module provides command-line access to corpus data operations,
particularly downloading external datasets like Rico.

Usage:
    python -m src.corpus download-rico --dataset semantic --data-dir ./data
    python -m src.corpus list-datasets
"""

import argparse
import sys
from pathlib import Path

from src.corpus.lib import (
    download_rico,
    get_data_dir,
    list_rico_datasets,
)


def cmd_download_rico(args: argparse.Namespace) -> int:
    """Handle the download-rico command."""
    try:
        dataset = download_rico(
            dataset_type=args.dataset,
            data_dir=args.data_dir,
            force=args.force,
        )
        print(f"\nSuccess! Dataset available at: {dataset.path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list_datasets(args: argparse.Namespace) -> int:
    """Handle the list-datasets command."""
    datasets = list_rico_datasets()
    print("Available Rico datasets:\n")
    for name, info in datasets.items():
        print(f"  {name}:")
        print(f"    Size: {info['size_mb']}MB")
        print(f"    Description: {info['description']}")
        print()
    return 0


def cmd_show_data_dir(args: argparse.Namespace) -> int:
    """Handle the data-dir command."""
    data_dir = get_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for corpus CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.corpus",
        description="Corpus data management for wireframe-mcp",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # download-rico command
    download_parser = subparsers.add_parser(
        "download-rico",
        help="Download Rico dataset",
    )
    download_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=["semantic", "ui_screenshots", "ui_metadata", "ui_vectors"],
        default="semantic",
        help="Dataset type to download (default: semantic)",
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
    download_parser.set_defaults(func=cmd_download_rico)

    # list-datasets command
    list_parser = subparsers.add_parser(
        "list-datasets",
        help="List available datasets",
    )
    list_parser.set_defaults(func=cmd_list_datasets)

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

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
