"""CLI entry point for wireframe-mcp.

This module acts as the central entry point for the project's CLI tools.
It delegates commands to the appropriate submodules or runs specific tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config import (
    EnvVar,
    get_available_llm_providers,
    get_environment,
    get_kroki_url,
)
from src.core import get_logger, setup_logging
from src.corpus.api import CorpusManager
from src.docker.cli import handle_docker_command

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

    # If --docker flag, re-execute inside container
    if getattr(args, "docker", False):
        from src.config import get_data_dir, get_index_dir
        from src.docker.exec import run_in_container

        # Build the command without --docker to avoid recursion
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

        # Mount corpus directories
        host_data = str(get_data_dir())
        host_index = str(get_index_dir().parent)  # Mount parent .corpus dir

        logger.info(f"Executing in Docker container: {args.image}")
        logger.info(f"Command: {' '.join(inner_cmd)}")

        result = run_in_container(
            command=inner_cmd,
            image=args.image,
            gpu=True,
            volumes={
                host_data: "/app/corpus/data",
                host_index: "/app/corpus",
            },
            env={"EMBEDDING_BACKEND": args.backend or "local"},
        )
        return result.returncode

    from src.config import get_default_embedding_backend, get_index_dir
    from src.vector import BackendType, VectorStore

    try:
        # Resolve output path
        output_path = args.output if args.output else get_index_dir()

        # Determine backend (auto-select if not specified)
        backend_name = args.backend or get_default_embedding_backend()
        backend_type = BackendType[backend_name.upper()]

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
        default=None,
        choices=["voyage", "local"],
        help="Embedding backend (default: auto-select based on VOYAGE_API_KEY)",
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
    build_parser.add_argument(
        "--docker",
        action="store_true",
        help="Execute inside Docker container with GPU support",
    )
    build_parser.add_argument(
        "--image",
        type=str,
        default="wireframe-mcp:latest",
        help="Docker image to use (default: wireframe-mcp:latest)",
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


# =============================================================================
# Stats Command (Corpus Data Profiling)
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
        kroki_url = get_kroki_url()
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


# =============================================================================
# Service Command (MCP Infrastructure Management)
# =============================================================================


def _check_llm_availability() -> dict:
    """Check which LLM providers are available.

    Returns dict with:
        - available: List of available provider names
        - preferred: The recommended provider to use (or None)
        - details: Dict of provider -> status info
    """
    import httpx

    results = {
        "available": [],
        "preferred": None,
        "details": {},
    }

    # Get cloud providers with available API keys from centralized config
    cloud_descriptions = {
        "openai": "OpenAI (GPT-4.1-mini)",
        "anthropic": "Anthropic (Claude Sonnet 4.5)",
        "deepseek": "DeepSeek (V3.2)",
        "qwen": "Qwen (Turbo)",
    }

    available_providers = get_available_llm_providers()
    for provider in available_providers:
        results["available"].append(provider)
        results["details"][provider] = {
            "type": "cloud",
            "description": cloud_descriptions.get(provider, provider),
            "status": "API key found",
        }
        if results["preferred"] is None:
            results["preferred"] = provider

    # Check for Ollama (local)
    ollama_url = get_environment(EnvVar.OLLAMA_HOST)
    try:
        response = httpx.get(f"{ollama_url}/api/tags", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            results["available"].append("ollama")
            results["details"]["ollama"] = {
                "type": "local",
                "description": f"Ollama ({len(models)} models)",
                "status": "Running",
                "models": models[:5],  # First 5 models
            }
            if results["preferred"] is None:
                results["preferred"] = "ollama"
    except Exception:
        results["details"]["ollama"] = {
            "type": "local",
            "description": "Ollama (local)",
            "status": "Not running",
        }

    return results


def _check_kroki_health(url: str, timeout: float = 5.0) -> bool:
    """Check if Kroki service is healthy."""
    import httpx

    try:
        response = httpx.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _check_voyage_api(timeout: float = 10.0) -> tuple[bool, str]:
    """Check if Voyage API is accessible for embeddings."""
    api_key = get_environment(EnvVar.VOYAGE_API_KEY)
    if not api_key:
        return False, "VOYAGE_API_KEY not set"

    try:
        import httpx

        response = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": "test", "model": "voyage-2"},
            timeout=timeout,
        )
        if response.status_code == 200:
            return True, "OK"
        return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {e}"


def _parse_init_args(args: list[str]) -> dict:
    """Parse service init arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python . service init",
        description="Initialize MCP server with first-time setup",
    )
    parser.add_argument(
        "--embedding",
        "-e",
        type=str,
        choices=["voyage", "local"],
        default="local",
        help="Embedding backend: voyage (cloud) or local (default: local)",
    )
    parser.add_argument(
        "--llm",
        "-l",
        type=str,
        choices=["openai", "anthropic", "deepseek", "qwen", "ollama"],
        default=None,
        help="Preferred LLM provider (auto-detected if not specified)",
    )
    parser.add_argument(
        "--kroki-port",
        type=int,
        default=18000,
        help="Port for Kroki rendering service (default: 18000)",
    )
    parser.add_argument(
        "--corpus",
        "-c",
        type=str,
        default="rico_semantic",
        help="Corpus provider to download (default: rico_semantic)",
    )
    parser.add_argument(
        "--index-limit",
        type=int,
        default=None,  # no limit by default
        help="Max items to index (default: no limit for --docker, 500 for local)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker service startup",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip corpus download and indexing",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker container for GPU-accelerated indexing",
    )
    parser.add_argument(
        "--show-env",
        action="store_true",
        help="Show MCP-compatible environment variables",
    )

    return vars(parser.parse_args(args))


def _show_mcp_env_config() -> None:
    """Display MCP-compatible environment configuration."""
    print()
    print("=" * 60)
    print("MCP Server Environment Configuration")
    print("=" * 60)
    print()
    print("Add these to your MCP client configuration (claude_desktop_config.json):")
    print()
    print("""{
  "mcpServers": {
    "wireframe": {
      "command": "python",
      "args": ["."],
      "cwd": "/path/to/wireframe-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "DEEPSEEK_API_KEY": "sk-...",
        "VOYAGE_API_KEY": "pa-...",
        "KROKI_URL": "http://localhost:18000",
        "KROKI_PORT": "18000",
        "MCP_PORT": "18080",
        "CORPUS_DATA_DIR": ".corpus/data",
        "INDEX_DIR": ".corpus/index",
        "EMBEDDING_BACKEND": "local",
        "LLM_PROVIDER": "openai"
      }
    }
  }
}""")
    print()
    print("Environment Variables:")
    print("  LLM Providers (at least one required):")
    print("    OPENAI_API_KEY      - OpenAI API key")
    print("    ANTHROPIC_API_KEY   - Anthropic API key")
    print("    DEEPSEEK_API_KEY    - DeepSeek API key")
    print("    QWEN_API_KEY        - Qwen/Alibaba API key")
    print("    OLLAMA_HOST         - Ollama server URL (default: localhost:11434)")
    print()
    print("  Embeddings:")
    print("    VOYAGE_API_KEY      - Voyage AI key (for cloud embeddings)")
    print("    EMBEDDING_BACKEND   - 'voyage' or 'local' (default: local)")
    print()
    print("  Services:")
    print("    KROKI_URL           - Kroki render service URL")
    print("    KROKI_PORT          - Kroki port (default: 18000)")
    print("    MCP_PORT            - MCP server port (default: 18080)")
    print()
    print("  Data:")
    print("    CORPUS_DATA_DIR     - Corpus data directory")
    print("    INDEX_DIR           - Vector index directory")
    print()


def cmd_service_init(args: list[str]) -> int:
    """Initialize MCP server - complete first-time setup.

    Steps:
    1. Check LLM availability (fail if none available)
    2. Start Docker services (with Kroki)
    3. Wait for health checks
    4. Download corpus data
    5. Build RAG index
    6. Verify optional services
    """
    import os
    import time

    from docker import get_compose_files

    # Parse arguments
    opts = _parse_init_args(args)

    # Show env config and exit if requested
    if opts["show_env"]:
        _show_mcp_env_config()
        return 0

    print("=" * 60)
    print("MCP Server Initialization")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Embedding backend: {opts['embedding']}")
    print(f"  Kroki port: {opts['kroki_port']}")
    print(f"  Corpus: {opts['corpus']}")
    print(f"  Docker indexing: {opts['docker']}")
    # Show effective index limit
    if opts["index_limit"] is not None:
        print(f"  Index limit: {opts['index_limit']}")
    elif opts["docker"]:
        print("  Index limit: no limit (docker)")
    else:
        print("  Index limit: 500 (local default)")
    if opts["llm"]:
        print(f"  Preferred LLM: {opts['llm']}")
    print()

    # Step 1: Check LLM availability
    print("[1/6] Checking LLM availability...")
    llm_status = _check_llm_availability()

    # Override preferred if specified
    if opts["llm"]:
        if opts["llm"] in llm_status["available"]:
            llm_status["preferred"] = opts["llm"]
        else:
            print(f"  WARNING: Requested LLM '{opts['llm']}' not available")

    if not llm_status["available"]:
        print()
        print("ERROR: No LLM provider available!")
        print()
        print("You need at least one of:")
        print("  - OPENAI_API_KEY (recommended)")
        print("  - ANTHROPIC_API_KEY")
        print("  - DEEPSEEK_API_KEY")
        print("  - Ollama running locally")
        print()
        print("Set an API key in your .env file or start Ollama.")
        print()
        print("For MCP configuration help, run:")
        print("  python . service init --show-env")
        return 1

    print(f"  Found {len(llm_status['available'])} provider(s):")
    for provider in llm_status["available"]:
        detail = llm_status["details"][provider]
        marker = "*" if provider == llm_status["preferred"] else " "
        print(f"  {marker} {detail['description']} - {detail['status']}")

    print(f"  Preferred: {llm_status['preferred']}")
    print()

    # Step 2: Start Docker services
    if opts["skip_docker"]:
        print("[2/6] Skipping Docker services (--skip-docker)")
    else:
        print("[2/6] Starting Docker services...")
        # Set port via environment
        os.environ["KROKI_PORT"] = str(opts["kroki_port"])

        try:
            compose_files = get_compose_files(mode="dev", include_kroki=True)
            compose_args = []
            for f in compose_files:
                compose_args.extend(["-f", str(f)])

            result = subprocess.run(
                ["docker", "compose", *compose_args, "up", "-d"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("  ERROR: Failed to start services")
                print(f"  {result.stderr}")
                return 1
            print("  Services started")
        except FileNotFoundError:
            print("  ERROR: Docker not found. Please install Docker.")
            return 1

    # Step 3: Wait for Kroki health
    kroki_url = get_kroki_url()
    if opts["skip_docker"]:
        print("[3/6] Skipping Kroki health check (--skip-docker)")
    else:
        print("[3/6] Waiting for Kroki service...")
        max_wait = 30
        for i in range(max_wait):
            if _check_kroki_health(kroki_url):
                print(f"  Kroki healthy at {kroki_url}")
                break
            time.sleep(1)
            if i % 5 == 4:
                print(f"  Waiting... ({i + 1}s)")
        else:
            print(f"  WARNING: Kroki not responding after {max_wait}s")
            print("  Rendering features may not work")
    print()

    # Step 4: Download corpus
    if opts["skip_corpus"]:
        print("[4/6] Skipping corpus download (--skip-corpus)")
    else:
        print("[4/6] Checking corpus data...")
        manager = CorpusManager()
        provider_name = opts["corpus"]

        try:
            provider = manager.get_provider(provider_name)
            if provider.has_data():
                print(f"  {provider_name} data already available")
            else:
                print(f"  Downloading {provider_name}...")
                provider.fetch(force=False)
                print("  Download complete")
        except KeyError:
            print(f"  ERROR: Unknown corpus provider '{provider_name}'")
            print(f"  Available: {', '.join(manager.list_providers())}")
        except Exception as e:
            print(f"  WARNING: Could not fetch corpus: {e}")
            print("  RAG features may not work")
    print()

    # Step 5: Build index
    if opts["skip_corpus"]:
        print("[5/6] Skipping index build (--skip-corpus)")
    else:
        print("[5/6] Checking RAG index...")
        from src.config import get_data_dir, get_index_dir

        index_dir = get_index_dir()
        index_file = index_dir / "index.faiss"

        if index_file.exists():
            print(f"  Index already exists at {index_dir}")
        else:
            embedding_backend = opts["embedding"]

            # Docker indexing path
            if opts["docker"]:
                print("  Building index with Docker (GPU-accelerated)...")
                try:
                    from src.docker.exec import run_in_container

                    # Build the command
                    inner_cmd = ["python", ".", "index", "build", "--all"]
                    if opts["index_limit"] is not None:
                        inner_cmd.extend(["--limit", str(opts["index_limit"])])
                    inner_cmd.extend(["--backend", embedding_backend])

                    # Mount corpus volumes
                    host_data = str(get_data_dir())
                    host_index = str(index_dir.parent)  # Mount parent .corpus dir

                    result = run_in_container(
                        command=inner_cmd,
                        image="wireframe-mcp:latest",
                        gpu=True,
                        volumes={
                            host_data: "/app/corpus/data",
                            host_index: "/app/corpus",
                        },
                        env={"EMBEDDING_BACKEND": embedding_backend},
                    )

                    if result.returncode == 0:
                        print("  Docker indexing complete")
                    else:
                        print("  WARNING: Docker indexing failed")
                        print("  Falling back to local indexing")
                        opts["docker"] = False  # Fall through to local indexing
                except Exception as e:
                    print(f"  WARNING: Docker indexing failed ({e})")
                    print("  Falling back to local indexing")
                    opts["docker"] = False  # Fall through to local indexing

            # Local indexing path (also used as fallback from docker)
            if not opts["docker"]:
                # Use sensible default limit of 500 for local if not specified
                limit = opts["index_limit"]
                index_limit = limit if limit is not None else 500

                print(f"  Building index with {embedding_backend} embeddings...")
                try:
                    from src.vector import BackendType, VectorStore

                    backend_type = (
                        BackendType.VOYAGE
                        if embedding_backend == "voyage"
                        else BackendType.LOCAL
                    )

                    # Check Voyage API if selected
                    if backend_type == BackendType.VOYAGE:
                        voyage_ok, voyage_msg = _check_voyage_api()
                        if not voyage_ok:
                            print(f"  WARNING: Voyage not available ({voyage_msg})")
                            print("  Falling back to local embeddings")
                            backend_type = BackendType.LOCAL

                    store = VectorStore(backend=backend_type)
                    indexed = 0

                    manager = CorpusManager()
                    for item in manager.stream_data(opts["corpus"]):
                        if item.layout is not None:
                            store.index_item(item)
                            indexed += 1
                            if indexed >= index_limit:
                                break

                    if indexed > 0:
                        store.save(index_dir)
                        print(f"  Indexed {indexed} items to {index_dir}")
                    else:
                        print("  WARNING: No items to index")
                except Exception as e:
                    print(f"  WARNING: Could not build index: {e}")
    print()

    # Step 6: Verify optional services
    print("[6/6] Checking optional services...")

    # Voyage API
    voyage_ok, voyage_msg = _check_voyage_api()
    if voyage_ok:
        print("  Voyage API: OK (production embeddings available)")
    else:
        print(f"  Voyage API: {voyage_msg} (using local embeddings)")

    print()
    print("=" * 60)
    print("Initialization Complete!")
    print("=" * 60)
    print()
    print("Quick start:")
    print("  python . generate 'login form with email and password'")
    print()
    print("Service management:")
    print("  python . service status   # Check health")
    print("  python . service stop     # Stop services")
    print("  python . service logs     # View logs")
    print()
    print("MCP configuration:")
    print("  python . service init --show-env")
    print()

    return 0


def cmd_service_status(_args: list[str]) -> int:
    """Check health status of all MCP services."""
    print("MCP Service Status")
    print("=" * 50)
    print()

    # LLM providers
    print("LLM Providers:")
    llm_status = _check_llm_availability()
    if llm_status["available"]:
        for provider in llm_status["available"]:
            detail = llm_status["details"][provider]
            print(f"  [OK] {detail['description']}")
        print(f"  Preferred: {llm_status['preferred']}")
    else:
        print("  [FAIL] No LLM provider available")
    print()

    # Kroki
    print("Rendering Service:")
    kroki_url = get_kroki_url()
    if _check_kroki_health(kroki_url):
        print(f"  [OK] Kroki at {kroki_url}")
    else:
        print(f"  [FAIL] Kroki not responding at {kroki_url}")
    print()

    # Voyage API
    print("Embedding Service:")
    voyage_ok, voyage_msg = _check_voyage_api()
    if voyage_ok:
        print("  [OK] Voyage API (cloud)")
    else:
        print(f"  [WARN] Voyage: {voyage_msg}")
        print("  [OK] Local embeddings available as fallback")
    print()

    # RAG Index
    print("RAG Index:")
    from src.config import get_index_dir

    index_dir = get_index_dir()
    index_file = index_dir / "index.faiss"
    if index_file.exists():
        import json

        # Read metadata directly to avoid backend initialization issues
        try:
            store_meta_path = index_dir / "index.store.json"
            if store_meta_path.exists():
                with open(store_meta_path) as f:
                    store_data = json.load(f)
                    item_count = len(store_data.get("metadata", {}))
                    backend = store_data.get("backend", "unknown")
                print(f"  [OK] {item_count} documents indexed (backend: {backend})")
            else:
                print(f"  [WARN] Index metadata not found at {store_meta_path}")
        except Exception as e:
            print(f"  [WARN] Index exists but failed to read: {e}")
    else:
        print(f"  [WARN] No index found at {index_dir}")
        print("  Run 'python . service init' to build index")
    print()

    return 0


def cmd_service(extra_args: list[str]) -> int:
    """Manage MCP server services.

    Usage:
        python . service init              # First-time setup
        python . service start [service]   # Start services (or specific service)
        python . service stop [service]    # Stop services (or specific service)
        python . service status            # Health check
        python . service logs [service]    # View logs
        python . service build [--no-cache] # Build images
    """
    import os

    from docker import (
        KROKI_PORT,
        MCP_PORT,
        get_compose_files,
        get_service_info,
        list_backends,
        list_modes,
        list_services,
    )

    # Build available services from docker module
    available_services = [get_service_info(s).container_name for s in list_services()]

    if not extra_args:
        print("MCP Service Management")
        print("\nUsage: python . service {command} [options] [service]")
        print("\nCommands:")
        print("  init                    First-time setup (recommended)")
        print("  start [service]         Start all services or specific service")
        print("  stop [service]          Stop all services or specific service")
        print("  status                  Health check all services")
        print("  logs [service]          View service logs")
        print("  build [--no-cache]      Build Docker images")
        print("  ps                      List running containers")
        print("\nOptions:")
        print("  --prod                  Use production configuration")
        print("  --no-kroki              Start without Kroki renderer")
        print("  --no-cache              Force rebuild without cache")
        print("  --kroki-port PORT       Override Kroki port (default: 18000)")
        print("  --mcp-port PORT         Override MCP server port (default: 18080)")
        print(f"\nServices: {', '.join(available_services)}")
        print("\nExamples:")
        print("  python . service init                   # Complete first-time setup")
        print("  python . service start                  # Start all services")
        print("  python . service start wfmcp-kroki      # Start only Kroki")
        print("  python . service stop wfmcp-server      # Stop only MCP server")
        print("  python . service build --no-cache       # Force rebuild images")
        print("  python . service logs wfmcp-kroki       # View Kroki logs")
        print(f"\nDefault ports: MCP={MCP_PORT}, Kroki={KROKI_PORT}")
        return 1

    command = extra_args[0]
    rest_args = extra_args[1:]

    # High-level commands
    if command == "init":
        return cmd_service_init(rest_args)

    if command == "status":
        return cmd_service_status(rest_args)

    # Info commands
    if command == "modes":
        logger.info("Available modes:")
        for mode in list_modes():
            logger.info(f"  - {mode}")
        return 0

    if command == "backends":
        logger.info("Available backends:")
        for backend in list_backends():
            logger.info(f"  - {backend}")
        return 0

    if command == "services":
        from docker import list_services_by_category

        logger.info("Available services:")
        for category, services in list_services_by_category().items():
            if services:
                logger.info(f"\n  {category.value.upper()}:")
                for svc in services:
                    info = get_service_info(svc)
                    logger.info(f"    {info.container_name} - {info.description}")
        return 0

    # Map service commands to docker compose
    compose_cmd_map = {
        "start": "up",
        "stop": "down",
        "build": "build",
        "logs": "logs",
        "ps": "ps",
        "restart": "restart",
    }

    if command in compose_cmd_map:
        try:
            # Parse flags and extract service targets
            mode = "dev"
            include_kroki = command == "start"  # Default kroki for start
            compose_args_extra = []
            no_cache = False
            target_services = []
            kroki_port = None
            mcp_port = None

            i = 0
            while i < len(rest_args):
                arg = rest_args[i]
                if arg == "--prod":
                    mode = "prod"
                elif arg == "--kroki":
                    include_kroki = True
                elif arg == "--no-kroki":
                    include_kroki = False
                elif arg in ("-d", "--detach"):
                    compose_args_extra.append("-d")
                elif arg == "--no-cache":
                    no_cache = True
                elif arg == "--kroki-port" and i + 1 < len(rest_args):
                    kroki_port = rest_args[i + 1]
                    i += 1
                elif arg == "--mcp-port" and i + 1 < len(rest_args):
                    mcp_port = rest_args[i + 1]
                    i += 1
                elif arg in available_services:
                    target_services.append(arg)
                elif not arg.startswith("-"):
                    # Could be a service name for logs
                    target_services.append(arg)
                else:
                    compose_args_extra.append(arg)
                i += 1

            # Set port environment variables if specified
            if kroki_port:
                os.environ["KROKI_PORT"] = kroki_port
            if mcp_port:
                os.environ["MCP_PORT"] = mcp_port

            # Get compose files
            compose_files = get_compose_files(mode=mode, include_kroki=include_kroki)
            compose_args = []
            for file in compose_files:
                compose_args.extend(["-f", str(file)])

            # Build final command
            docker_cmd = compose_cmd_map[command]
            final_cmd = ["docker", "compose", *compose_args, docker_cmd]

            # Add command-specific options
            if command == "start":
                final_cmd.append("-d")
            if command == "build" and no_cache:
                final_cmd.append("--no-cache")

            final_cmd.extend(compose_args_extra)

            # Add target services at the end
            final_cmd.extend(target_services)

            # Log what we're doing
            mode_str = f"Mode: {mode}" + (" + kroki" if include_kroki else "")
            if kroki_port or mcp_port:
                ports_str = []
                if mcp_port:
                    ports_str.append(f"MCP={mcp_port}")
                if kroki_port:
                    ports_str.append(f"Kroki={kroki_port}")
                mode_str += f" (ports: {', '.join(ports_str)})"

            logger.info(mode_str)
            if target_services:
                logger.info(f"Target: {', '.join(target_services)}")
            logger.info(f"Running: {' '.join(final_cmd)}")

            try:
                return subprocess.call(final_cmd)
            except KeyboardInterrupt:
                return 130

        except Exception as e:
            logger.error(f"Error: {e}")
            return 1

    else:
        logger.error(f"Unknown service command: {command}")
        print("\nRun 'python . service' for usage information")
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
        print("  corpus     Corpus data management (download, list)")
        print("  demo       Render pipeline demonstration")
        print("\nExamples:")
        print("  python . dev test --unit           # Fast unit tests")
        print("  python . dev test --integration    # Integration tests")
        print("  python . dev stats rico_semantic   # Profile provider data")
        print("  python . dev benchmark -v          # Run benchmarks verbose")
        print("  python . dev corpus download rico  # Download corpus")
        print("  python . dev demo login -r         # Render login demo")
        return 1

    subcommand = argv[0]
    subargs = argv[1:]

    dev_commands = {
        "test": lambda: cmd_test(subargs),
        "stats": lambda: handle_stats_command(subargs),
        "benchmark": lambda: handle_benchmark_command(subargs),
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


# =============================================================================
# MCP Server Command
# =============================================================================


def handle_mcp_command(argv: list[str]) -> int:
    """Handle MCP server commands.

    Usage:
        python . mcp run              # Start in STDIO mode (for Claude Desktop)
        python . mcp serve            # Start in HTTP mode
        python . mcp serve --port 8080
        python . mcp info             # Show server info
    """
    if not argv:
        print("MCP Server Commands")
        print("\nUsage: python . mcp {command} [options]")
        print("\nCommands:")
        print("  run                 Start server in STDIO mode (for Claude Desktop)")
        print("  serve               Start server in HTTP mode")
        print("  info                Show server information")
        print("\nOptions for 'serve':")
        print("  --host HOST         Bind address (default: 0.0.0.0)")
        print("  --port PORT         Port number (default: 18080)")
        print("  --transport TYPE    Transport: http or sse (default: http)")
        print("\nExamples:")
        print("  python . mcp run                    # STDIO for Claude Desktop")
        print("  python . mcp serve                  # HTTP on port 18080")
        print("  python . mcp serve --port 8080     # HTTP on custom port")
        print("\nClaude Desktop Configuration:")
        print("  Add to claude_desktop_config.json:")
        print("  {")
        print('    "mcpServers": {')
        print('      "wireframe": {')
        print('        "command": "python",')
        print('        "args": [".", "mcp", "run"],')
        print('        "cwd": "/path/to/wireframe-mcp"')
        print("      }")
        print("    }")
        print("  }")
        return 1

    subcommand = argv[0]
    subargs = argv[1:]

    if subcommand == "run":
        # STDIO mode for Claude Desktop
        from src.mcp.server import TransportType, run_server

        logger.info("Starting MCP server in STDIO mode...")
        run_server(transport=TransportType.STDIO)
        return 0

    elif subcommand == "serve":
        # HTTP/SSE mode for web deployment
        from src.mcp.server import TransportType, run_server

        # Parse serve arguments
        host = "0.0.0.0"
        port = 18080
        transport = TransportType.HTTP

        i = 0
        while i < len(subargs):
            arg = subargs[i]
            if arg == "--host" and i + 1 < len(subargs):
                host = subargs[i + 1]
                i += 2
            elif arg == "--port" and i + 1 < len(subargs):
                port = int(subargs[i + 1])
                i += 2
            elif arg == "--transport" and i + 1 < len(subargs):
                transport = TransportType(subargs[i + 1])
                i += 2
            else:
                i += 1

        logger.info(f"Starting MCP server in {transport.value} mode...")
        logger.info(f"Listening on {host}:{port}")
        run_server(transport=transport, host=host, port=port)
        return 0

    elif subcommand == "info":
        # Show server information
        from src.mcp import get_server_capabilities, get_server_version

        print("Wireframe MCP Server")
        print("=" * 40)
        print(f"Version: {get_server_version()}")
        print("\nCapabilities:")
        for cap, enabled in get_server_capabilities().items():
            status = "enabled" if enabled else "disabled"
            print(f"  {cap}: {status}")
        print("\nAvailable Tools:")
        print("  - ping: Health check")
        print("  - get_server_info: Server metadata")
        print("\nPlanned Tools (Phase 2):")
        print("  - generate_layout: Generate UI layouts")
        print("  - search_layouts: Search vector database")
        print("  - render_layout: Render to PNG/SVG")
        print("  - validate_layout: Validate structure")
        print("  - transpile_layout: Convert to DSL")
        return 0

    else:
        logger.error(f"Unknown mcp command: {subcommand}")
        return handle_mcp_command([])


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
    print("\n=== Development ===")
    print("  dev        Development workflows (test, benchmark, etc.)")
    print("\nGetting Started:")
    print("  python . service init               # First-time setup")
    print("  python . service status             # Check service health")
    print("  python . docker up                  # Start Docker stack")
    print("\nMCP Server:")
    print("  python . mcp run                    # Start STDIO server (Claude Desktop)")
    print("  python . mcp serve                  # Start HTTP server on port 18080")
    print("  python . mcp info                   # Show server information")
    print("\nLayout Generation:")
    print("  python . generate 'login form with email and password'")
    print("  python . generate models")
    print("  python . search 'dashboard with sidebar' -k 3")
    print("\nService Management:")
    print("  python . service start              # Start services")
    print("  python . service stop               # Stop services")
    print("  python . docker ps                  # Check Docker services")
    print("\nIndex Management:")
    print("  python . index build --all          # Build RAG index from all providers")
    print("  python . index build rico_semantic  # Build index from specific provider")
    print("  python . index info .corpus/index   # Show index information")
    print("\nDevelopment Examples:")
    print("  python . dev test --unit             # Run unit tests")
    print("  python . dev stats                   # Profile corpus data")
    print("  python . dev benchmark -v            # Run benchmarks")
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
        "mcp": lambda: handle_mcp_command(rest_args),
        "generate": lambda: handle_generate_command(rest_args),
        "search": lambda: handle_search_command(rest_args),
        "service": lambda: cmd_service(rest_args),
        "docker": lambda: handle_docker_command(rest_args),
        "index": lambda: handle_index_command(rest_args),
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
