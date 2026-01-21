# GPU-Accelerated RAG Index Build via Docker CLI

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable full GPU-accelerated RAG index builds via `python . index` CLI commands that execute inside Docker containers with FAISS-GPU.

**Architecture:**
- Add `python . index` as a top-level CLI command (not under `dev`)
- Support `--docker` flag to execute commands inside GPU-enabled container
- Fix Dockerfile to install faiss-gpu-cu12 instead of relying on conda
- Remove hardcoded limits from service init, provide sensible defaults

**Tech Stack:** Docker, FAISS-GPU, CUDA 12.6, Python 3.12, argparse CLI

---

## Problem Analysis

### Current Issues

1. **FAISS-GPU not in Dockerfile** - The Dockerfile installs PyTorch with CUDA but not faiss-gpu
2. **Windows has no faiss-gpu pip wheels** - Only faiss-cpu available via pip on Windows
3. **Index commands buried under `dev`** - Should be top-level for production use
4. **No `--docker` execution mode** - Cannot run CLI commands inside container from host
5. **service init has 100-item default limit** - Too small for quality assessment
6. **No full corpus index workflow** - Need end-to-end GPU-accelerated build

### Solution Overview

1. Update Dockerfile to install `faiss-gpu-cu12` for Linux container
2. Add `python . index` top-level command group
3. Add `--docker` flag to execute index commands inside container
4. Create `python . index build --all --docker` workflow
5. Fix service init limits and defaults

---

## Task 1: Add faiss-gpu-cu12 to Dockerfile

**Files:**
- Modify: `docker/Dockerfile:50-55`

**Step 1: Write the test for FAISS GPU in container**

```bash
# Test command to verify FAISS GPU after build
docker run --gpus all wireframe-mcp:dev python -c "
import faiss
print('FAISS version:', faiss.__version__)
print('Has GPU:', hasattr(faiss, 'StandardGpuResources'))
print('Num GPUs:', faiss.get_num_gpus())
"
```

**Step 2: Update Dockerfile builder stage**

In `docker/Dockerfile`, after line 53 (torch installation), add faiss-gpu-cu12:

```dockerfile
# Install Python dependencies with GPU support
# CRITICAL: PyTorch MUST be installed with CUDA from PyTorch's index BEFORE sentence-transformers
# This ensures GPU acceleration is available for embeddings and model inference
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir faiss-gpu-cu12 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .
```

**Step 3: Update requirements.txt comments**

Replace the FAISS comments (lines 11-13):

```
# Vector Database & Math
# FAISS GPU: Installed in Docker via pip (faiss-gpu-cu12)
# Windows native: Use faiss-cpu (no GPU wheels available)
# Linux native: pip install faiss-gpu-cu12
numpy
```

**Step 4: Build and verify**

```bash
cd docker
docker compose -f compose.base.yml -f compose.core.yml build wfmcp-server
docker run --gpus all wireframe-mcp:latest python -c "import faiss; print(faiss.get_num_gpus())"
# Expected: 1 (or more)
```

**Step 5: Commit**

```bash
git add docker/Dockerfile requirements.txt
git commit -m "feat(docker): add faiss-gpu-cu12 for GPU-accelerated indexing"
```

---

## Task 2: Add top-level `python . index` command group

**Files:**
- Modify: `__main__.py:2314` (command dispatch)
- Modify: `__main__.py:504-580` (move from dev subcommand)

**Step 1: Update command dispatch table**

In `__main__.py`, find the dispatch table (around line 2314) and add index:

```python
commands = {
    "mcp": lambda: handle_mcp_command(rest_args),
    "generate": lambda: handle_generate_command(rest_args),
    "search": lambda: handle_search_command(rest_args),
    "service": lambda: handle_service_command(rest_args),
    "docker": lambda: handle_docker_command(rest_args),
    "index": lambda: handle_index_command(rest_args),  # ADD THIS
    "dev": lambda: handle_dev_command(rest_args),
}
```

**Step 2: Update help text**

In the main usage help (around line 2280), add index section:

```python
=== Index Management ===
  index      Build and manage RAG vector indices
```

**Step 3: Run test**

```bash
python . index --help
# Should show: build, info subcommands
python . index build --help
# Should show: provider, --all, --limit, --docker flags
```

**Step 4: Commit**

```bash
git add __main__.py
git commit -m "feat(cli): promote index command to top-level"
```

---

## Task 3: Add `--docker` execution mode to index commands

**Files:**
- Modify: `__main__.py` (handle_index_command function)
- Create: `src/docker/exec.py` (container execution helper)

**Step 1: Write failing test for docker exec helper**

Create `tests/docker/test_exec.py`:

```python
"""Tests for Docker container execution."""
import pytest
from src.docker.exec import build_docker_exec_command

def test_build_exec_command_basic():
    """Test basic command building."""
    cmd = build_docker_exec_command(
        command=["python", ".", "index", "build", "--all"],
        container="wfmcp-server",
    )
    assert cmd == [
        "docker", "exec", "wfmcp-server",
        "python", ".", "index", "build", "--all"
    ]

def test_build_exec_command_with_gpu():
    """Test GPU-enabled run command."""
    cmd = build_docker_exec_command(
        command=["python", ".", "index", "build"],
        container=None,  # Use run instead of exec
        image="wireframe-mcp:latest",
        gpu=True,
    )
    assert "--gpus" in cmd
    assert "all" in cmd
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/docker/test_exec.py -v
# Expected: FAIL - module not found
```

**Step 3: Create src/docker/exec.py**

```python
"""Docker container execution helpers.

Provides utilities for running CLI commands inside Docker containers,
with support for GPU passthrough and volume mounts.
"""

import subprocess
from pathlib import Path


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
        ValueError: If neither container nor image specified.
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/docker/test_exec.py -v
# Expected: PASS
```

**Step 5: Update src/docker/__init__.py**

```python
from .exec import build_docker_exec_command, run_in_container

__all__ = [
    # ... existing exports ...
    "build_docker_exec_command",
    "run_in_container",
]
```

**Step 6: Commit**

```bash
git add src/docker/exec.py tests/docker/test_exec.py src/docker/__init__.py
git commit -m "feat(docker): add container execution helpers"
```

---

## Task 4: Wire --docker flag into index build command

**Files:**
- Modify: `__main__.py` (handle_index_command, cmd_index_build)

**Step 1: Add --docker argument to build parser**

In `handle_index_command`, find the build_parser section and add:

```python
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
```

**Step 2: Update cmd_index_build to handle --docker**

At the start of `cmd_index_build` function, add:

```python
def cmd_index_build(args: argparse.Namespace) -> int:
    """Handle the index build command."""

    # If --docker flag, re-execute inside container
    if getattr(args, "docker", False):
        from src.docker.exec import run_in_container
        from src.config import get_data_dir, get_index_dir

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

    # Original implementation continues here...
```

**Step 3: Test the workflow**

```bash
# Build image first
python . docker build

# Test docker execution
python . index build rico_semantic --docker --limit 100
# Should show: Executing in Docker container...
# Should complete with GPU acceleration
```

**Step 4: Commit**

```bash
git add __main__.py
git commit -m "feat(cli): add --docker flag for GPU-accelerated index builds"
```

---

## Task 5: Fix service init defaults and limits

**Files:**
- Modify: `__main__.py` (parse_service_init_args, cmd_service_init)

**Step 1: Change default --index-limit**

In `parse_service_init_args` (around line 1520), change:

```python
parser.add_argument(
    "--index-limit",
    type=int,
    default=None,  # Changed from 100 - no limit by default
    help="Max items to index during init (default: no limit for --docker, 500 for local)",
)
```

**Step 2: Add --docker flag to service init**

```python
parser.add_argument(
    "--docker",
    action="store_true",
    help="Use Docker container for GPU-accelerated indexing",
)
```

**Step 3: Update indexing logic in cmd_service_init**

In the indexing section of `cmd_service_init`, update to use docker when requested:

```python
# [5/6] Build RAG index
if not opts["skip_corpus"]:
    print("\n[5/6] Building RAG index...")

    if opts.get("docker"):
        # Use GPU-accelerated Docker build
        from src.docker.exec import run_in_container

        build_cmd = ["python", ".", "index", "build", "--all"]
        if opts["index_limit"]:
            build_cmd.extend(["--limit", str(opts["index_limit"])])

        print(f"  Running GPU-accelerated build in Docker...")
        result = run_in_container(
            command=build_cmd,
            image="wireframe-mcp:latest",
            gpu=True,
            volumes={...},
        )
        if result.returncode != 0:
            print("  [WARN] Docker index build failed, falling back to local")
    else:
        # Original local indexing with sensible default limit
        limit = opts["index_limit"] or 500  # Default for local
        # ... existing code ...
```

**Step 4: Test**

```bash
# Test local init (should default to 500)
python . service init --skip-docker --skip-corpus
python . service init --skip-docker --index-limit 1000

# Test docker init (should use GPU, no default limit)
python . service init --docker
```

**Step 5: Commit**

```bash
git add __main__.py
git commit -m "fix(cli): improve service init defaults, add --docker flag"
```

---

## Task 6: Create full index build workflow

**Files:**
- Create: `scripts/build-full-index.sh` (convenience script)
- Modify: Documentation

**Step 1: Create convenience script**

Create `scripts/build-full-index.sh`:

```bash
#!/bin/bash
# Build complete RAG index with GPU acceleration
#
# Usage: ./scripts/build-full-index.sh [--limit N]
#
# Prerequisites:
#   - Docker with NVIDIA runtime
#   - wireframe-mcp:latest image built
#   - Corpus data downloaded

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Full RAG Index Build (GPU-Accelerated)"
echo "=========================================="

# Check prerequisites
if ! docker info --format '{{.Runtimes}}' | grep -q nvidia; then
    echo "ERROR: NVIDIA Docker runtime not found"
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Build image if needed
if ! docker image inspect wireframe-mcp:latest &>/dev/null; then
    echo "Building Docker image..."
    cd "$PROJECT_ROOT"
    python . docker build
fi

# Run full index build
echo ""
echo "Starting GPU-accelerated index build..."
cd "$PROJECT_ROOT"
python . index build --all --docker "$@"

echo ""
echo "=========================================="
echo "Index build complete!"
echo "=========================================="
python . index info
```

**Step 2: Make executable and test**

```bash
chmod +x scripts/build-full-index.sh
./scripts/build-full-index.sh --limit 1000  # Test with limit first
./scripts/build-full-index.sh               # Full build
```

**Step 3: Update README with workflow**

Add to README.md:

```markdown
## GPU-Accelerated Index Build

For optimal performance, build the RAG index using GPU acceleration:

```bash
# One-time setup
python . docker build

# Full index build (GPU)
python . index build --all --docker

# Or use convenience script
./scripts/build-full-index.sh
```

Requirements:
- NVIDIA GPU with CUDA 12+
- Docker with NVIDIA runtime
- ~10GB disk space for full corpus
```

**Step 4: Commit**

```bash
git add scripts/build-full-index.sh README.md
git commit -m "docs: add GPU-accelerated full index build workflow"
```

---

## Task 7: Integration test

**Files:**
- Create: `tests/integration/test_gpu_index.py`

**Step 1: Write integration test**

```python
"""Integration tests for GPU-accelerated indexing."""
import subprocess
import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_nvidia_docker(),
    reason="NVIDIA Docker runtime not available"
)
def test_docker_index_build():
    """Test full Docker-based index build."""
    result = subprocess.run(
        ["python", ".", "index", "build", "rico_semantic",
         "--docker", "--limit", "50"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0
    assert "GPU" in result.stdout or "gpu" in result.stdout.lower()


def _has_nvidia_docker():
    """Check if NVIDIA Docker runtime is available."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
        )
        return "nvidia" in result.stdout
    except Exception:
        return False
```

**Step 2: Run integration test**

```bash
pytest tests/integration/test_gpu_index.py -v -m integration
```

**Step 3: Commit**

```bash
git add tests/integration/test_gpu_index.py
git commit -m "test: add GPU index integration tests"
```

---

## Summary

After completing all tasks:

| Command | Description |
|---------|-------------|
| `python . index build --all` | Build index locally (CPU) |
| `python . index build --all --docker` | Build index in container (GPU) |
| `python . index info` | Show index statistics |
| `python . service init --docker` | Full setup with GPU indexing |
| `./scripts/build-full-index.sh` | Convenience script for full GPU build |

The key changes:
1. Dockerfile now installs `faiss-gpu-cu12` for GPU support
2. `index` command promoted to top-level CLI
3. `--docker` flag enables container execution with GPU
4. Service init has sensible defaults (no arbitrary limits)
5. Full documentation and convenience scripts provided

---

**Plan complete and saved to `docs/plans/2026-01-21-gpu-indexing-cli.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
