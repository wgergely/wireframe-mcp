"""CLI entry point for wireframe-mcp.

This module acts as the central entry point for the project's CLI tools.
It delegates commands to the appropriate submodules or runs specific tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.core import get_logger, setup_logging
from src.corpus.api import CorpusManager

# Load environment variables from .env file
load_dotenv()

logger = get_logger("cli")


# =============================================================================
# Generate Command
# =============================================================================


def cmd_generate_layout(args: argparse.Namespace) -> int:
    """Handle the generate layout command."""
    from src.llm import (
        GeneratorConfig,
        LayoutGenerator,
        LLMModel,
        create_llm_backend,
    )

    try:
        # Create backend if model specified
        backend = None
        if args.model:
            model = LLMModel.by_name(args.model)
            if model is None:
                logger.error(f"Unknown model: {args.model}")
                logger.info("Available models:")
                for m in LLMModel:
                    logger.info(f"  {m.spec.name}")
                return 1
            backend = create_llm_backend(model, api_key=args.api_key)

        # Load vector store if index path provided
        vector_store = None
        if args.index:
            from src.vector import VectorStore

            vector_store = VectorStore()
            vector_store.load(args.index)
            logger.info(f"Loaded vector index from {args.index}")

        # Configure generator
        config = GeneratorConfig(
            temperature=args.temperature,
            max_retries=args.retries,
            include_rag_context=vector_store is not None,
        )

        # Generate layout
        generator = LayoutGenerator(
            backend=backend,
            vector_store=vector_store,
            config=config,
        )

        logger.info(f"Generating layout for: {args.query}")
        output = generator.generate(args.query, target_provider=args.target)

        # Output result based on format
        from src.output import OutputGenerator

        layout_json = output.context.node.model_dump_json(indent=2)
        output_gen = OutputGenerator(default_provider=args.target)
        layout_output = output_gen.generate_from_context(output.context)

        if args.format == "json":
            result_text = layout_json
        elif args.format == "tree":
            result_text = layout_output.text_tree
        elif args.format in ("d2", "plantuml"):
            result_text = layout_output.dsl_code
        elif args.format == "all":
            dsl = layout_output.dsl_code
            result_text = (
                f"## Text Tree\n{layout_output.text_tree}\n\n"
                f"## DSL ({args.target})\n```{args.target}\n{dsl}\n```\n\n"
                f"## JSON\n```json\n{layout_json}\n```"
            )
        else:
            result_text = layout_json

        if args.output:
            args.output.write_text(result_text)
            logger.info(f"Layout saved to {args.output}")
        else:
            print(result_text)

        # Show stats
        logger.info(
            f"Stats: {output.stats.attempts} attempt(s), "
            f"{output.stats.total_tokens} tokens, "
            f"model={output.stats.final_model}"
        )
        return 0

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


def cmd_list_models(_args: argparse.Namespace) -> int:
    """Handle the list models command."""
    from src.llm import LLMModel, LLMProviderType

    logger.info("Available LLM Models:")
    for provider in LLMProviderType:
        models = LLMModel.list_by_provider(provider)
        if models:
            logger.info(f"\n  {provider.value}:")
            for model in models:
                spec = model.spec
                local_tag = " (local)" if spec.is_local else ""
                logger.info(f"    {spec.name}{local_tag}")
    return 0


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
    layout_parser.add_argument(
        "query",
        type=str,
        help="Natural language description of the layout",
    )
    layout_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="LLM model name (e.g. gpt-4.1-mini, claude-sonnet-4-5)",
    )
    layout_parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=None,
        help="API key (uses env var if not provided)",
    )
    layout_parser.add_argument(
        "--index",
        "-i",
        type=Path,
        default=None,
        help="Vector index path for RAG context",
    )
    layout_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (prints to stdout if not specified)",
    )
    layout_parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="d2",
        choices=["d2", "plantuml"],
        help="Target transpiler (default: d2)",
    )
    layout_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    layout_parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max generation retries (default: 3)",
    )
    layout_parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="json",
        choices=["json", "tree", "d2", "plantuml", "all"],
        help="Output format (default: json)",
    )
    layout_parser.set_defaults(func=cmd_generate_layout)

    # models command
    models_parser = subparsers.add_parser(
        "models",
        help="List available LLM models",
    )
    models_parser.set_defaults(func=cmd_list_models)

    # Default to layout if query provided directly
    if argv and not argv[0].startswith("-") and argv[0] not in ("layout", "models"):
        argv = ["layout"] + argv

    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


# =============================================================================
# Index Command
# =============================================================================


def _build_index_for_provider(
    manager: CorpusManager,
    provider_name: str,
    store,  # VectorStore - avoid import at module level
    batch_size: int,
    limit: int | None,
    skip_download: bool,
) -> tuple[int, int]:
    """Build index for a single provider with auto-download.

    Args:
        manager: CorpusManager instance.
        provider_name: Name of the provider to index.
        store: VectorStore instance to add documents to.
        batch_size: Batch size for embedding.
        limit: Optional limit on documents to index.
        skip_download: Skip download even if data missing.

    Returns:
        Tuple of (items_indexed, items_skipped).
    """
    logger.info("")
    logger.info(f"{'=' * 60}")
    logger.info(f"Provider: {provider_name}")
    logger.info(f"{'=' * 60}")

    try:
        provider = manager.get_provider(provider_name)
    except KeyError as e:
        logger.error(f"  Provider not found: {e}")
        return 0, 0

    # Check if data exists, download if needed
    if not provider.has_data():
        if skip_download:
            logger.warning("  Data not available, skipping (--skip-download)")
            return 0, 0

        logger.info("  Data not found locally, downloading...")
        try:
            provider.fetch(force=False)
            logger.info("  Download complete")
        except Exception as e:
            logger.error(f"  Download failed: {e}")
            return 0, 0
    else:
        logger.info("  Data already available locally")

    # Count items with layouts
    logger.info("  Scanning for items with layouts...")
    items_with_layout = 0
    items_without_layout = 0

    items_to_index = []
    for item in provider.process():
        if item.layout is not None:
            items_to_index.append(item)
            items_with_layout += 1
            if limit and items_with_layout >= limit:
                break
        else:
            items_without_layout += 1

    logger.info(f"  Found {items_with_layout} items with layouts")
    if items_without_layout > 0:
        logger.info(f"  Skipped {items_without_layout} items without layouts")

    if items_with_layout == 0:
        logger.warning("  No items to index, skipping provider")
        return 0, items_without_layout

    # Index items with progress
    logger.info(f"  Indexing {len(items_to_index)} items...")

    indexed_count = 0
    for i, item in enumerate(items_to_index):
        try:
            store.index_item(item)
            indexed_count += 1

            # Progress every 100 items or at completion
            if (i + 1) % 100 == 0 or (i + 1) == len(items_to_index):
                pct = ((i + 1) / len(items_to_index)) * 100
                logger.info(f"    Progress: {i + 1}/{len(items_to_index)} ({pct:.1f}%)")
        except Exception as e:
            logger.warning(f"    Failed to index item {item.id}: {e}")

    logger.info(f"  Indexed {indexed_count} items from {provider_name}")
    return indexed_count, items_without_layout


def cmd_build_index(args: argparse.Namespace) -> int:
    """Handle the build index command."""
    from src.config import get_index_dir
    from src.vector import BackendType, VectorStore

    try:
        # Resolve output path
        output_path = args.output if args.output else get_index_dir()

        # Determine backend
        backend_type = BackendType[args.backend.upper()]

        logger.info("")
        logger.info("=" * 60)
        logger.info("RAG Index Builder")
        logger.info("=" * 60)
        logger.info(f"Backend: {backend_type.value}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Batch size: {args.batch_size}")
        if args.limit:
            logger.info(f"Limit per provider: {args.limit}")

        # Create corpus manager
        manager = CorpusManager()
        available_providers = manager.list_providers()

        # Determine which providers to index
        if args.all:
            providers_to_index = available_providers
            logger.info(f"Mode: ALL providers ({len(providers_to_index)} total)")
        elif args.provider:
            if args.provider not in available_providers:
                logger.error(f"Unknown provider: {args.provider}")
                logger.info(f"Available providers: {', '.join(available_providers)}")
                return 1
            providers_to_index = [args.provider]
            logger.info(f"Mode: Single provider ({args.provider})")
        else:
            logger.error("Must specify either a provider name or --all")
            return 1

        logger.info(f"Providers to index: {', '.join(providers_to_index)}")

        # Create vector store
        store = VectorStore(backend=backend_type)

        # Track overall progress
        total_indexed = 0
        total_skipped = 0
        successful_providers = []
        failed_providers = []

        # Index each provider
        for i, provider_name in enumerate(providers_to_index, 1):
            logger.info("")
            logger.info(
                f"[{i}/{len(providers_to_index)}] Processing {provider_name}..."
            )

            try:
                indexed, skipped = _build_index_for_provider(
                    manager=manager,
                    provider_name=provider_name,
                    store=store,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    skip_download=args.skip_download,
                )
                total_indexed += indexed
                total_skipped += skipped

                if indexed > 0:
                    successful_providers.append(provider_name)
                else:
                    failed_providers.append(provider_name)

            except KeyboardInterrupt:
                logger.warning("")
                logger.warning("Interrupted by user")
                logger.info("Saving partial index...")
                break
            except Exception as e:
                logger.error(f"Error processing {provider_name}: {e}")
                failed_providers.append(provider_name)
                continue

        # Save index
        logger.info("")
        logger.info("=" * 60)
        logger.info("Saving Index")
        logger.info("=" * 60)

        if total_indexed > 0:
            store.save(output_path)
            logger.info(f"Index saved to: {output_path}")
        else:
            logger.warning("No items indexed, skipping save")

        # Final summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Build Summary")
        logger.info("=" * 60)
        logger.info(f"Total items indexed: {total_indexed}")
        logger.info(f"Total items skipped: {total_skipped}")
        logger.info(f"Successful providers: {len(successful_providers)}")
        if successful_providers:
            for p in successful_providers:
                logger.info(f"  - {p}")
        logger.info(f"Failed/empty providers: {len(failed_providers)}")
        if failed_providers:
            for p in failed_providers:
                logger.info(f"  - {p}")

        if total_indexed > 0:
            stats = store.stats()
            logger.info("")
            logger.info("Index Statistics:")
            logger.info(f"  Total documents: {stats.total_items}")
            logger.info(f"  Embedding dimension: {stats.dimension}")
            logger.info(f"  Backend: {stats.embedding_backend}")
            logger.info(f"  GPU enabled: {stats.is_gpu}")

        logger.info("")
        logger.info("Build complete!")
        return 0

    except KeyError:
        logger.error(f"Unknown backend: {args.backend}")
        logger.info("Available backends: voyage, local")
        return 1
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_index_info(args: argparse.Namespace) -> int:
    """Handle the index info command."""
    from src.vector import VectorStore

    try:
        store = VectorStore()
        store.load(args.path)
        stats = store.stats()

        logger.info(f"Index path: {args.path}")
        logger.info(f"Total documents: {stats.total_items}")
        logger.info(f"Embedding dimension: {stats.dimension}")
        logger.info(f"Backend: {stats.embedding_backend}")
        logger.info(f"GPU enabled: {stats.is_gpu}")

        if stats.sources:
            logger.info("Sources:")
            for source, count in sorted(stats.sources.items()):
                logger.info(f"  {source}: {count}")

        return 0

    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return 1


def handle_index_command(argv: list[str]) -> int:
    """Handle index-specific commands."""
    parser = argparse.ArgumentParser(
        prog="python . index",
        description="Build and manage vector indices",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build a vector index from corpus data",
    )
    build_parser.add_argument(
        "provider",
        type=str,
        nargs="?",
        default=None,
        help="Corpus provider name (e.g. rico_semantic). Use --all for all.",
    )
    build_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Build index from all available providers",
    )
    build_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for index (default: .corpus/index)",
    )
    build_parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="voyage",
        choices=["voyage", "local"],
        help="Embedding backend (default: voyage)",
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing (default: 100)",
    )
    build_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents per provider",
    )
    build_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip providers that need downloading (use existing data only)",
    )
    build_parser.set_defaults(func=cmd_build_index)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about an index",
    )
    info_parser.add_argument(
        "path",
        type=Path,
        help="Path to the index directory",
    )
    info_parser.set_defaults(func=cmd_index_info)

    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


# =============================================================================
# Search Command
# =============================================================================


def cmd_search_index(args: argparse.Namespace) -> int:
    """Handle the search command."""
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

    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)
    return cmd_search_index(args)


# =============================================================================
# Benchmark Command
# =============================================================================


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run search quality benchmark against the vector index."""
    from src.config import get_data_dir
    from src.vector import VectorStore
    from src.vector.audit import (
        AuditRunner,
        calculate_metrics,
    )
    from src.vector.backend import create_backend

    # Resolve index path
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = get_data_dir() / args.index

    faiss_file = index_path.with_suffix(".faiss")
    if not faiss_file.exists():
        logger.error(f"Index not found: {faiss_file}")
        return 1

    logger.info(f"Loading index from {index_path}...")

    # Create backend and load store
    try:
        backend = create_backend(args.model)
    except ValueError as e:
        logger.error(f"Backend error: {e}")
        return 1

    store = VectorStore(backend=backend, index_path=index_path)
    stats = store.stats()
    logger.info(f"Index loaded: {stats.total_items} items, {stats.dimension}D")

    # Define benchmark queries
    queries = _get_benchmark_queries(args.queries)

    if not queries:
        logger.error("No benchmark queries defined")
        return 1

    logger.info(f"Running {len(queries)} benchmark queries...")

    # Run audit
    def on_complete(result):
        status = "PASS" if result.top_result_passed else "FAIL"
        logger.info(f"  [{status}] {result.query.name}: {result.query_time_ms:.1f}ms")

    runner = AuditRunner(store, on_query_complete=on_complete if args.verbose else None)
    report = runner.run(queries)

    # Calculate metrics
    k_values = [1, 3, 5, 10] if stats.total_items >= 10 else [1, 3, 5]
    metrics = calculate_metrics(report, k_values=k_values)

    # Print results
    print("\n" + "=" * 50)
    print(metrics.summary())
    print("=" * 50)
    print(
        f"\nPass Rate: {report.pass_rate:.1f}% "
        f"({report.passed_queries}/{report.total_queries})"
    )
    print(f"Avg Query Time: {report.average_query_time_ms:.2f}ms")

    if report.average_best_rank is not None:
        print(f"Avg Best Match Rank: {report.average_best_rank:.2f}")

    return 0 if report.pass_rate >= args.min_pass_rate else 1


def _get_benchmark_queries(queries_file: str | None) -> list:
    """Load or generate benchmark queries."""
    from src.vector.audit import (
        create_component_query,
    )

    if queries_file:
        # TODO: Load from JSON file
        logger.warning(f"Custom queries file not yet supported: {queries_file}")

    # Default benchmark queries for UI layout search
    return [
        create_component_query(
            query="login form with email and password inputs",
            name="login_form",
            components=["input", "button"],
            required_components=["input"],
        ),
        create_component_query(
            query="navigation bar with menu items",
            name="navbar",
            components=["toolbar", "button"],
        ),
        create_component_query(
            query="dashboard with charts and statistics",
            name="dashboard",
            components=["container", "image", "text"],
        ),
        create_component_query(
            query="settings page with toggles and checkboxes",
            name="settings",
            components=["checkbox", "button"],
        ),
        create_component_query(
            query="user profile with avatar and information",
            name="profile",
            components=["image", "text"],
        ),
        create_component_query(
            query="search bar with input field",
            name="search_bar",
            components=["input"],
            required_components=["input"],
        ),
        create_component_query(
            query="modal dialog with buttons",
            name="modal",
            components=["container", "button"],
        ),
        create_component_query(
            query="list view with items",
            name="list_view",
            components=["container"],
        ),
        create_component_query(
            query="form with multiple input fields",
            name="multi_input_form",
            components=["input", "button"],
            required_components=["input"],
        ),
        create_component_query(
            query="card layout with image and text",
            name="card",
            components=["card", "image", "text"],
        ),
    ]


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

    args = parser.parse_args(argv)
    return cmd_benchmark(args)


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
        help="Target data directory (default: .corpus/data)",
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
    """Run pytest with provided arguments and test tier options.

    Usage:
        python . test                # Run all tests
        python . test --unit         # Run only unit tests (fast, no dependencies)
        python . test --integration  # Run integration tests (file/network, no Docker)
        python . test --docker       # Run Docker-dependent tests (requires Kroki)
        python . test --all          # Run all tests explicitly
        python . test -v             # Run with verbose output
        python . test -k "search"    # Run tests matching pattern

    Test Tiers:
        unit        - Fast tests with no I/O or external dependencies
        integration - Tests requiring file system, network, or external APIs
        docker      - Tests requiring Docker services (Kroki, etc.)
        rag         - Tests requiring vector index (auto-built if missing)
    """
    # Test tier mapping
    tier_markers = {
        "--unit": ["-m", "unit"],
        "--integration": ["-m", "integration and not docker"],
        "--docker": ["-m", "docker or kroki"],
        "--rag": ["-m", "rag"],
        "--all": [],  # No filter, run everything
    }

    # Process tier flags
    pytest_args: list[str] = []
    remaining_args: list[str] = []

    for arg in extra_args:
        if arg in tier_markers:
            pytest_args.extend(tier_markers[arg])
        else:
            remaining_args.append(arg)

    # Build final command
    cmd = [sys.executable, "-m", "pytest", *pytest_args, *remaining_args]
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def cmd_build(extra_args: list[str]) -> int:
    """Run docker compose commands.

    Usage:
        python . build              # Build dev image
        python . build up           # Start dev containers
        python . build down         # Stop containers
        python . build --prod up    # Start production containers
        python . build --prod build # Build production image
        python . build --kroki up   # Start with kroki rendering backend
    """
    from docker import get_compose_files

    try:
        # Parse flags
        mode = "dev"
        include_kroki = False

        # Process flags
        remaining_args = []
        for arg in extra_args:
            if arg == "--prod":
                mode = "prod"
            elif arg == "--kroki":
                include_kroki = True
            else:
                remaining_args.append(arg)

        # Get compose files
        compose_files = get_compose_files(mode=mode, include_kroki=include_kroki)

        # Build command
        action = remaining_args if remaining_args else ["build"]
        compose_args = []
        for file in compose_files:
            compose_args.extend(["-f", str(file)])

        final_cmd = ["docker", "compose", *compose_args, *action]

        logger.info(f"Mode: {mode}")
        if include_kroki:
            logger.info("Backends: kroki rendering enabled")
        logger.info(f"Running: {' '.join(final_cmd)}")

        try:
            return subprocess.call(final_cmd)
        except KeyboardInterrupt:
            return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


# =============================================================================
# Audit Command
# =============================================================================


def cmd_audit_corpus(args: argparse.Namespace) -> int:
    """Run cross-provider corpus audit for data quality assessment."""
    from collections import Counter

    from src.corpus.normalizer import (
        count_components,
        extract_text_content,
        node_count,
        tree_depth,
    )
    from src.mid import ComponentType, is_valid

    manager = CorpusManager()
    providers = manager.list_providers()

    if args.provider:
        if args.provider not in providers:
            logger.error(f"Unknown provider: {args.provider}")
            logger.info(f"Available: {', '.join(providers)}")
            return 1
        providers = [args.provider]

    all_metrics = []
    limit = args.limit

    for provider_name in providers:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"AUDITING: {provider_name}")
        logger.info("=" * 60)

        try:
            provider = manager.get_provider(provider_name)
        except KeyError:
            logger.warning("  Provider not found, skipping")
            continue

        if not provider.has_data():
            logger.warning("  No data available, skipping")
            continue

        metrics = {
            "provider": provider_name,
            "total": 0,
            "has_layout": 0,
            "layout_valid": 0,
            "has_screenshot": 0,
            "has_text_content": 0,
            "component_counts": Counter(),
            "avg_tree_depth": 0,
            "avg_node_count": 0,
        }

        depths = []
        node_counts = []

        for i, item in enumerate(manager.stream_data(provider_name)):
            if i >= limit:
                break

            metrics["total"] += 1

            if item.layout is not None:
                metrics["has_layout"] += 1
                if is_valid(item.layout):
                    metrics["layout_valid"] += 1
                depths.append(tree_depth(item.layout))
                node_counts.append(node_count(item.layout))
                comp_counts = count_components(item.layout)
                for comp_type, count in comp_counts.items():
                    metrics["component_counts"][comp_type] += count
                texts = extract_text_content(item.layout)
                if texts:
                    metrics["has_text_content"] += 1

            if item.screenshot_path and item.screenshot_path.exists():
                metrics["has_screenshot"] += 1

        if depths:
            metrics["avg_tree_depth"] = sum(depths) / len(depths)
        if node_counts:
            metrics["avg_node_count"] = sum(node_counts) / len(node_counts)

        total = metrics["total"]
        if total > 0:
            logger.info(f"  Samples: {total}")
            logger.info(
                f"  Layout: {metrics['has_layout']}/{total} "
                f"({100 * metrics['has_layout'] / total:.0f}%)"
            )
            logger.info(
                f"  Valid: {metrics['layout_valid']}/{total} "
                f"({100 * metrics['layout_valid'] / total:.0f}%)"
            )
            logger.info(f"  Avg depth: {metrics['avg_tree_depth']:.1f}")
            logger.info(f"  Avg nodes: {metrics['avg_node_count']:.1f}")

            if metrics["component_counts"]:
                logger.info("  Top components:")
                for comp, count in metrics["component_counts"].most_common(5):
                    logger.info(f"    {comp}: {count}")

        all_metrics.append(metrics)

    # Cross-provider comparison (if multiple)
    if len(all_metrics) > 1:
        logger.info("")
        logger.info("=" * 80)
        logger.info("CROSS-PROVIDER COMPARISON")
        logger.info("=" * 80)

        known_types = set(ct.value for ct in ComponentType)
        all_components: Counter = Counter()
        for m in all_metrics:
            all_components.update(m["component_counts"])

        logger.info("")
        logger.info(
            f"{'Provider':<15} {'Samples':>8} {'Layout%':>8} "
            f"{'Valid%':>8} {'Depth':>6} {'Nodes':>6}"
        )
        logger.info("-" * 60)

        for m in all_metrics:
            total = m["total"]
            if total == 0:
                continue
            logger.info(
                f"{m['provider']:<15} {total:>8} "
                f"{100 * m['has_layout'] / total:>7.0f}% "
                f"{100 * m['layout_valid'] / total:>7.0f}% "
                f"{m['avg_tree_depth']:>6.1f} "
                f"{m['avg_node_count']:>6.1f}"
            )

        logger.info("")
        logger.info("Top component types across providers:")
        for comp, count in all_components.most_common(10):
            in_mid = "✓" if comp in known_types else "✗"
            logger.info(f"  {comp:<20} {count:>6} {in_mid}")

    return 0


def handle_audit_command(argv: list[str]) -> int:
    """Handle audit command argument parsing."""
    parser = argparse.ArgumentParser(
        prog="python . audit",
        description="Audit corpus data quality for vector DB and LLM grounding",
    )
    parser.add_argument(
        "provider",
        type=str,
        nargs="?",
        default=None,
        help="Specific provider to audit (default: all providers)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=50,
        help="Number of samples per provider (default: 50)",
    )

    args = parser.parse_args(argv)
    return cmd_audit_corpus(args)


# =============================================================================
# Demo Command
# =============================================================================


def cmd_demo_render(args: argparse.Namespace) -> int:
    """Demonstrate the render pipeline with sample layouts."""
    from src.mid import LayoutNode, Orientation
    from src.providers import get_provider
    from src.render import OutputFormat, RenderClient, RenderConfig
    from src.schema import ComponentType

    # Sample layouts
    layouts = {
        "dashboard": lambda: LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            label="Dashboard",
            orientation=Orientation.HORIZONTAL,
            children=[
                LayoutNode(
                    id="sidebar",
                    type=ComponentType.TOOLBAR,
                    label="Navigation",
                    orientation=Orientation.VERTICAL,
                    flex_ratio=3,
                    children=[
                        LayoutNode(id="logo", type=ComponentType.IMAGE, label="Logo"),
                        LayoutNode(
                            id="nav-home", type=ComponentType.BUTTON, label="Home"
                        ),
                        LayoutNode(
                            id="nav-settings",
                            type=ComponentType.BUTTON,
                            label="Settings",
                        ),
                    ],
                ),
                LayoutNode(
                    id="main",
                    type=ComponentType.CONTAINER,
                    label="Content",
                    flex_ratio=9,
                    children=[
                        LayoutNode(
                            id="header", type=ComponentType.TOOLBAR, label="Header"
                        ),
                        LayoutNode(
                            id="cards",
                            type=ComponentType.CONTAINER,
                            orientation=Orientation.HORIZONTAL,
                            children=[
                                LayoutNode(
                                    id="card1", type=ComponentType.CARD, label="Stats"
                                ),
                                LayoutNode(
                                    id="card2", type=ComponentType.CARD, label="Chart"
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        "login": lambda: LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            label="Login",
            orientation=Orientation.VERTICAL,
            align="center",
            justify="center",
            children=[
                LayoutNode(
                    id="form",
                    type=ComponentType.CARD,
                    label="Login Form",
                    children=[
                        LayoutNode(
                            id="title", type=ComponentType.TEXT, label="Welcome Back"
                        ),
                        LayoutNode(id="email", type=ComponentType.INPUT, label="Email"),
                        LayoutNode(
                            id="password", type=ComponentType.INPUT, label="Password"
                        ),
                        LayoutNode(
                            id="submit", type=ComponentType.BUTTON, label="Sign In"
                        ),
                    ],
                ),
            ],
        ),
        "settings": lambda: LayoutNode(
            id="root",
            type=ComponentType.CONTAINER,
            label="Settings",
            orientation=Orientation.VERTICAL,
            children=[
                LayoutNode(id="header", type=ComponentType.TOOLBAR, label="Settings"),
                LayoutNode(
                    id="options",
                    type=ComponentType.LIST_ITEM,
                    label="Options",
                    children=[
                        LayoutNode(
                            id="opt1",
                            type=ComponentType.CHECKBOX,
                            label="Enable notifications",
                        ),
                        LayoutNode(
                            id="opt2",
                            type=ComponentType.CHECKBOX,
                            label="Dark mode",
                        ),
                        LayoutNode(
                            id="opt3",
                            type=ComponentType.CHECKBOX,
                            label="Auto-save",
                        ),
                    ],
                ),
                LayoutNode(id="save", type=ComponentType.BUTTON, label="Save Changes"),
            ],
        ),
    }

    # Get layout
    layout_name = args.layout
    if layout_name not in layouts:
        logger.error(f"Unknown layout: {layout_name}")
        logger.info(f"Available layouts: {', '.join(layouts.keys())}")
        return 1

    layout = layouts[layout_name]()

    # Transpile
    provider = get_provider(args.provider)
    dsl = provider.transpile(layout)

    logger.info(f"Layout: {layout_name}")
    logger.info(f"Provider: {args.provider}")
    logger.info("")
    logger.info("DSL Output:")
    logger.info("-" * 40)
    print(dsl)
    logger.info("-" * 40)

    # Render if requested
    if args.render:
        import os

        kroki_url = os.environ.get("KROKI_URL", "http://localhost:8000")
        client = RenderClient(base_url=kroki_url)

        if not client.is_available():
            logger.error("Kroki service not available")
            logger.info("Start with: python . build --kroki up")
            return 1

        output_format = OutputFormat[args.format.upper()]
        config = RenderConfig(output_format=output_format)
        result = client.render(dsl, args.provider, config)

        output_path = args.output or Path(f"demo_{layout_name}.{args.format}")
        output_path.write_bytes(result.data)
        logger.info(f"Rendered to: {output_path} ({result.size_bytes} bytes)")

    return 0


def handle_demo_command(argv: list[str]) -> int:
    """Handle demo command argument parsing."""
    parser = argparse.ArgumentParser(
        prog="python . demo",
        description="Demonstrate the render pipeline with sample layouts",
    )
    parser.add_argument(
        "layout",
        type=str,
        nargs="?",
        default="dashboard",
        choices=["dashboard", "login", "settings"],
        help="Sample layout to render (default: dashboard)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="plantuml",
        choices=["d2", "plantuml"],
        help="DSL provider (default: plantuml)",
    )
    parser.add_argument(
        "--render",
        "-r",
        action="store_true",
        help="Render to image (requires Kroki)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="png",
        choices=["png", "svg"],
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path",
    )

    args = parser.parse_args(argv)
    return cmd_demo_render(args)


def cmd_docker(extra_args: list[str]) -> int:
    """Manage docker backends and configuration.

    Usage:
        python . docker modes              # List available modes
        python . docker backends           # List available backends
        python . docker compose [args]     # Run docker compose directly
    """
    from docker import get_compose_files, list_backends, list_modes

    if not extra_args:
        print("Docker backend management")
        print("\nUsage: python . docker {command}")
        print("\nCommands:")
        print("  modes              List available modes (dev, prod)")
        print("  backends           List available backends (kroki, etc)")
        print("  compose [args]     Run docker compose with all backends")
        return 1

    command = extra_args[0]
    rest_args = extra_args[1:]

    if command == "modes":
        logger.info("Available modes:")
        for mode in list_modes():
            logger.info(f"  - {mode}")
        return 0

    elif command == "backends":
        logger.info("Available backends:")
        for backend in list_backends():
            logger.info(f"  - {backend}")
        return 0

    elif command == "compose":
        # Run docker compose with all backends enabled
        try:
            mode = "dev"
            if "--prod" in rest_args:
                mode = "prod"
                rest_args = [arg for arg in rest_args if arg != "--prod"]

            compose_files = get_compose_files(mode=mode, include_kroki=True)
            compose_args = []
            for file in compose_files:
                compose_args.extend(["-f", str(file)])

            final_cmd = ["docker", "compose", *compose_args, *rest_args]
            logger.info(f"Running: {' '.join(final_cmd)}")

            try:
                return subprocess.call(final_cmd)
            except KeyboardInterrupt:
                return 130

        except Exception as e:
            logger.error(f"Error: {e}")
            return 1

    else:
        logger.error(f"Unknown docker command: {command}")
        return 1


# =============================================================================
# Dev Subcommand Handler
# =============================================================================


def handle_dev_command(argv: list[str]) -> int:
    """Handle development workflow commands.

    Usage:
        python . dev test [args]       # Run pytest
        python . dev stats [provider]  # Corpus data profiling
        python . dev benchmark [args]  # Performance benchmarks
        python . dev index [args]      # Build/manage indices
        python . dev corpus [args]     # Corpus data management
        python . dev demo [args]       # Render pipeline demo
    """
    if not argv:
        print("Development workflow commands")
        print("\nUsage: python . dev {command} [args]")
        print("\nCommands:")
        print("  test       Run pytest with tier options")
        print("  stats      Corpus data profiling and statistics")
        print("  benchmark  Search quality and performance benchmarks")
        print("  index      Build and manage vector indices")
        print("  corpus     Corpus data management (download, list)")
        print("  demo       Render pipeline demonstration")
        print("\nExamples:")
        print("  python . dev test --unit           # Fast unit tests")
        print("  python . dev test --integration    # Integration tests")
        print("  python . dev stats rico_semantic   # Profile provider data")
        print("  python . dev benchmark -v          # Run benchmarks verbose")
        print("  python . dev index build --all     # Build full index")
        print("  python . dev corpus download rico  # Download corpus")
        print("  python . dev demo login -r         # Render login demo")
        return 1

    subcommand = argv[0]
    subargs = argv[1:]

    dev_commands = {
        "test": lambda: cmd_test(subargs),
        "stats": lambda: handle_stats_command(subargs),
        "benchmark": lambda: handle_benchmark_command(subargs),
        "index": lambda: handle_index_command(subargs),
        "corpus": lambda: handle_corpus_command(subargs),
        "demo": lambda: handle_demo_command(subargs),
    }

    if subcommand in dev_commands:
        return dev_commands[subcommand]()

    logger.error(f"Unknown dev command: {subcommand}")
    return handle_dev_command([])  # Show help


def handle_stats_command(argv: list[str]) -> int:
    """Handle stats command (renamed from audit - data profiling)."""
    # Reuse the existing audit command implementation
    return handle_audit_command(argv)


def show_help() -> None:
    """Display CLI help message."""
    print("Usage: python . {command} [args]")
    print("\n=== MCP Server Operations ===")
    print("  generate   Generate UI layouts from natural language")
    print("  search     Search vector indices for similar layouts")
    print("  build      Manage Docker containers (start/stop services)")
    print("  docker     Docker backend configuration")
    print("\n=== Development ===")
    print("  dev        Development workflows (test, benchmark, etc.)")
    print("\nMCP Examples:")
    print("  python . generate 'login form with email and password'")
    print("  python . generate models")
    print("  python . search 'dashboard with sidebar' -k 3")
    print("  python . build --kroki up            # Start Kroki renderer")
    print("  python . build down                  # Stop services")
    print("\nDevelopment Examples:")
    print("  python . dev test --unit             # Run unit tests")
    print("  python . dev stats                   # Profile corpus data")
    print("  python . dev benchmark -v            # Run benchmarks")
    print("  python . dev index build --all       # Build RAG index")
    print("  python . dev corpus download rico    # Download corpus data")
    print("  python . dev demo dashboard          # Demo render pipeline")
    print("\nFor dev command details: python . dev")


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
        "generate": lambda: handle_generate_command(rest_args),
        "search": lambda: handle_search_command(rest_args),
        "build": lambda: cmd_build(rest_args),
        "docker": lambda: cmd_docker(rest_args),
    }

    # Development commands (nested under 'dev')
    if command == "dev":
        return handle_dev_command(rest_args)

    commands = mcp_commands

    if command in commands:
        setup_logging()
        return commands[command]()

    logger.error(f"Unknown command: {command}")
    show_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
