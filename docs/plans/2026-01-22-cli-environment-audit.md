# CLI Environment Execution Audit

**Date**: 2026-01-22
**Branch**: `feature/cli-env-audit`
**Scope**: Full CLI architecture audit - Docker vs local execution handling

## Executive Summary

The wireframe-mcp CLI (`__main__.py`) has grown organically with **inconsistent environment execution patterns**. Currently:

- **Only `index build --docker`** explicitly supports Docker-based execution
- **All other commands** implicitly assume local execution
- **No unified `--env` flag pattern** exists across commands
- **CUDA/FAISS dependencies** create platform-specific failure modes on Windows

This audit documents the current state, identifies gaps, and provides a foundation for designing a unified environment execution system.

---

## 1. CLI Command Structure Overview

### 1.1 Top-Level Commands

| Command | Handler Function | Docker Support | GPU Dependencies |
|---------|------------------|----------------|------------------|
| `mcp` | `handle_mcp_command()` | None | Indirect (via vector search) |
| `generate` | `handle_generate_command()` | None | Indirect (if RAG enabled) |
| `search` | `handle_search_command()` | None | **Yes** (FAISS load) |
| `service` | `cmd_service()` | Orchestration only | Indirect |
| `docker` | `handle_docker_command()` | **Yes** (full) | N/A (manages containers) |
| `index` | `handle_index_command()` | **Partial** (`build --docker`) | **Yes** (FAISS, PyTorch) |
| `dev` | `handle_dev_command()` | None | Varies by subcommand |

### 1.2 Command Hierarchy

```
python .
├── mcp
│   ├── run                 # STDIO mode (Claude Desktop)
│   ├── serve               # HTTP mode
│   └── info                # Server info
├── generate
│   ├── layout <query>      # Generate from NL
│   └── models              # List LLM models
├── search <query>          # Search vector index
├── service
│   ├── init                # First-time setup
│   ├── start               # Start Docker services
│   ├── stop                # Stop Docker services
│   ├── status              # Health check
│   ├── logs                # View logs
│   └── build               # Build images
├── docker
│   ├── up                  # Start services
│   ├── down                # Stop services
│   ├── ps                  # List containers
│   ├── build               # Build images
│   ├── exec/run/logs/...   # Proxy commands
├── index
│   ├── build [--docker]    # Build RAG index (ONLY command with --docker!)
│   └── info                # Index stats
└── dev
    ├── test                # Run pytest
    ├── stats               # Corpus profiling
    ├── benchmark           # Performance tests
    ├── corpus              # Data management
    └── demo                # Render demo
```

---

## 2. Docker vs Local Execution Analysis

### 2.1 Current Docker Execution Pattern

**Only one command** currently supports explicit Docker execution: `index build --docker`

```python
# __main__.py:337-380 - cmd_build_index()
if getattr(args, "docker", False):
    from src.docker.exec import run_in_container

    inner_cmd = [".", "index", "build"]
    # ... build command without --docker to avoid recursion

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
```

**Key observation**: The `--docker` flag is command-specific, not a global pattern.

### 2.2 Docker Execution Infrastructure

**Location**: `src/docker/exec.py`

```python
def run_in_container(
    command: list[str],
    container: str | None = None,  # For docker exec (running container)
    image: str | None = None,      # For docker run (new container)
    gpu: bool = False,
    volumes: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    workdir: str = "/app",
) -> subprocess.CompletedProcess:
```

**Capabilities**:
- GPU passthrough (`--gpus all`)
- Volume mounts
- Environment variable injection
- Works with both `docker exec` (running container) and `docker run` (ephemeral container)

### 2.3 Commands That Should Support Docker Execution

| Command | Why Docker? | Current State |
|---------|-------------|---------------|
| `index build` | FAISS-GPU, PyTorch CUDA | **Has `--docker` flag** |
| `search` | FAISS index loading | No Docker support |
| `benchmark` | FAISS queries | No Docker support |
| `generate layout --index` | RAG context uses FAISS | No Docker support |
| `dev stats` | May load embeddings | No Docker support |
| `mcp run` | Full server with all features | Uses `service` for Docker |

---

## 3. GPU/CUDA Dependency Mapping

### 3.1 Direct FAISS Dependencies

**File**: `src/vector/index.py`

```python
class FAISSIndex:
    def _detect_gpu(self) -> bool:
        import faiss
        has_gpu_support = hasattr(faiss, "StandardGpuResources")
        num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        # Returns True only if faiss-gpu is installed AND GPU detected
```

**Import chain**:
```
__main__.py
  → cmd_build_index()
    → VectorStore()
      → FAISSIndex()
        → import faiss  # FAILS on Windows without faiss-cpu
```

### 3.2 PyTorch/sentence-transformers Dependencies

**File**: `src/vector/backend/local.py`

```python
class LocalBackend:
    def _get_model(self) -> SentenceTransformer:
        # Uses sentence-transformers which depends on PyTorch
        # GPU acceleration auto-detected by PyTorch
```

**Import chain**:
```
__main__.py
  → cmd_build_index()
    → VectorStore(backend=BackendType.LOCAL)
      → LocalBackend()
        → sentence_transformers  # Requires PyTorch
          → torch  # Requires CUDA for GPU
```

### 3.3 Platform-Specific Installation Notes

**From `requirements.txt`**:
```
# FAISS GPU: Installed in Docker via pip (faiss-gpu-cu12)
# Windows native: Use faiss-cpu (no GPU wheels available)
# Linux native: pip install faiss-gpu-cu12

# PyTorch with CUDA 12.4 support
# Install GPU version: pip install torch --index-url https://download.pytorch.org/whl/cu124
# Install CPU version: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3.4 Commands Affected by GPU Dependencies

| Command | FAISS Required | PyTorch Required | Breaks on Windows (GPU) |
|---------|----------------|------------------|-------------------------|
| `index build` | Yes | Yes (if local backend) | Yes |
| `index info` | Yes (load index) | No | Partial |
| `search` | Yes | Yes (for embedding query) | Yes |
| `benchmark` | Yes | Yes | Yes |
| `generate --index` | Yes | Yes | Yes |
| `mcp serve` (with RAG) | Yes | Yes | Yes |

---

## 4. Identified Gaps and Inconsistencies

### 4.1 Gap: No Unified Environment Flag

**Problem**: `--docker` only exists on `index build`. Other commands that need GPU have no way to request Docker execution.

**Impact**: Users on Windows cannot run `search`, `benchmark`, or RAG-enabled generation without manually coordinating Docker.

**Proposed Solution**: Add global `--env {local|docker}` flag or per-command `--docker` flags consistently.

### 4.2 Gap: Implicit Local venv Assumption

**Problem**: All commands implicitly assume they're running in a local venv with all dependencies.

**Impact**: No clear error messages when dependencies fail. Users see cryptic import errors.

**Example**:
```bash
python . search "login form"
# ImportError: cannot import name 'StandardGpuResources' from 'faiss'
```

**Proposed Solution**: Add dependency checks at command start with helpful error messages suggesting Docker alternative.

### 4.3 Inconsistency: Two Docker Command Systems

**Problem**: Both `docker` command and `service` command manage Docker, with overlapping functionality.

```
python . docker up       # Low-level compose wrapper
python . service start   # High-level with health checks
```

**Impact**: Confusion about which to use. Documentation inconsistency.

**Observation**: This is intentional layering (low-level vs high-level) but the relationship should be clearer.

### 4.4 Gap: No Local venv Execution Mode for Docker-First Commands

**Problem**: `service init --docker` can build index in Docker, but there's no inverse - no way to explicitly request local execution with dependency validation.

**Proposed Solution**: Add `--env local` that validates dependencies before execution.

### 4.5 Inconsistency: Backend Selection Scattered

**Problem**: Embedding backend selection (`--backend voyage|local`) exists in some commands but not others.

| Command | Backend Selection |
|---------|-------------------|
| `index build` | `--backend voyage\|local` |
| `search` | Reads from saved index metadata |
| `generate` | No option (uses default) |
| `service init` | `--embedding voyage\|local` |

### 4.6 Gap: No Environment Introspection Command

**Problem**: No command to check if local environment can run GPU-dependent operations.

**Proposed Solution**: Add `python . env check` or similar that validates:
- FAISS availability (cpu vs gpu)
- PyTorch CUDA availability
- Embedding backend availability
- Docker availability

---

## 5. Execution Path Diagrams

### 5.1 `index build` Execution Flow

```
User: python . index build rico --docker

                    __main__.py
                         │
                         ▼
              handle_index_command()
                         │
                         ▼
                  cmd_build_index()
                         │
                    ┌────┴────┐
                    │ --docker │
                    │   flag?  │
                    └────┬────┘
               Yes ◄────┴────► No
                │               │
                ▼               ▼
        run_in_container()   VectorStore()
                │               │
                ▼               ▼
          docker run         LocalBackend()
          --gpus all             │
          wireframe-mcp          ▼
          . index build    sentence_transformers
          rico                   │
                                 ▼
                           FAISSIndex()
                                 │
                                 ▼
                         import faiss
                              (FAILS on Windows
                               without faiss-cpu)
```

### 5.2 `search` Execution Flow (No Docker Option)

```
User: python . search "login form"

                    __main__.py
                         │
                         ▼
              handle_search_command()
                         │
                         ▼
                  cmd_search_index()
                         │
                         ▼
                    VectorStore()
                         │
                         ▼
                  FAISSIndex.load()
                         │
                         ▼
                    import faiss
                         │
                   (NO --docker option)
                   (FAILS on Windows GPU)
```

### 5.3 `service init` Execution Flow

```
User: python . service init --docker

                    __main__.py
                         │
                         ▼
                   cmd_service()
                         │
                         ▼
               cmd_service_init()
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   Check LLM       Start Docker    Build Index
   Providers       Services        (if --docker)
          │              │              │
          ▼              ▼              ▼
   get_available_  docker compose   run_in_container()
   llm_providers()    up -d              │
                                         ▼
                                   . index build
                                   --all --backend local
```

---

## 6. Module Dependency Graph

### 6.1 GPU-Dependent Module Chain

```
                    CLI Commands
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   index build       search          generate
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                    VectorStore
                    (src/vector/lib.py)
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        FAISSIndex           EmbeddingBackend
   (src/vector/index.py)   (src/vector/backend/)
              │                     │
              │              ┌──────┴──────┐
              │              │             │
              ▼              ▼             ▼
         import faiss   LocalBackend  VoyageBackend
              │              │             │
              │              ▼             │
              │      sentence_transformers │
              │              │             │
              │              ▼             │
              │          PyTorch           │
              │              │             │
              └──────────────┼─────────────┘
                             │
                    GPU Dependencies:
                    - faiss-gpu (CUDA)
                    - torch (CUDA)
```

### 6.2 Non-GPU Module Chain

```
                    CLI Commands
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
     corpus          generate          demo
    commands         (no RAG)        commands
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       CorpusManager          LLM Backend
   (src/corpus/api/lib.py)  (src/llm/backend/)
              │                     │
              ▼                     │
         Providers                  │
    (rico, enrico, etc.)           │
              │                     │
              ▼                     ▼
         HTTP/File I/O        OpenAI/Anthropic API
              │                     │
              └──────────────┬──────┘
                             │
                    No GPU Dependencies
                    (Pure Python + HTTP)
```

---

## 7. Configuration Layer Analysis

### 7.1 Environment Variable Management

**Central config**: `src/config/lib.py`

```python
class EnvVar(Enum):
    # Embedding config
    EMBEDDING_BACKEND = EnvConfig(name="EMBEDDING_BACKEND", default="local", ...)
    VECTOR_USE_GPU = EnvConfig(name="VECTOR_USE_GPU", default=None, ...)  # None = auto-detect

    # Corpus paths
    CORPUS_DATA_DIR = EnvConfig(name="CORPUS_DATA_DIR", default=None, ...)
    CORPUS_INDEX_DIR = EnvConfig(name="CORPUS_INDEX_DIR", default=None, ...)
```

**Docker container env** (from `docker/Dockerfile`):
```dockerfile
ENV EMBEDDING_BACKEND=local \
    VECTOR_USE_GPU=true \
    CORPUS_DATA_DIR=/app/corpus/data \
    CORPUS_INDEX_DIR=/app/corpus/index
```

### 7.2 Path Resolution Differences

| Context | CORPUS_DATA_DIR | CORPUS_INDEX_DIR |
|---------|-----------------|------------------|
| Local (default) | `.corpus/data` | `.corpus/index` |
| Docker container | `/app/corpus/data` | `/app/corpus/index` |
| Docker volume | Mounted to `/app/corpus/data` | Mounted to `/app/corpus/index` |

**Gap**: When running `--docker`, volume paths must be explicitly mapped. This is handled in `cmd_build_index` but not generalized.

---

## 8. Recommendations Summary

### 8.1 High Priority

1. **Add dependency validation** at CLI entry point with clear error messages
2. **Add `--docker` flag** to `search`, `benchmark` commands
3. **Create `env check` command** for environment introspection

### 8.2 Medium Priority

4. **Unify backend selection** - consider global `--embedding-backend` flag
5. **Document `docker` vs `service` command relationship** clearly
6. **Add platform-aware installation docs** (Windows vs Linux vs Docker)

### 8.3 Low Priority (Simplification)

7. **Consider consolidating** `docker` and `service` commands
8. **Consider global `--env` flag** vs per-command `--docker`
9. **Add venv validation mode** for local execution

---

## 9. Appendix: File Locations

### 9.1 Key Files for Environment Handling

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point, command routing |
| `src/docker/exec.py` | `run_in_container()` utility |
| `src/docker/cli.py` | `docker` command group |
| `docker/lib.py` | Docker constants, compose file selection |
| `src/config/lib.py` | Environment variable management |
| `src/vector/index.py` | FAISS wrapper with GPU detection |
| `src/vector/backend/local.py` | sentence-transformers backend |

### 9.2 Docker Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Multi-stage build (base, builder, dev, prod) |
| `docker/compose.base.yml` | Networks, volumes |
| `docker/compose.core.yml` | MCP server service |
| `docker/compose.renderer.yml` | Kroki services |
| `docker/compose.hotreload.yml` | Dev overrides |
| `docker/compose.production.yml` | Prod overrides |

---

## 10. Next Steps

1. Review this audit with stakeholders
2. Prioritize gaps based on user impact
3. Create implementation plan for chosen improvements
4. Consider creating separate design document for unified environment execution system
