# Reset Command Design

## Overview

Add a `reset` CLI command to clear development environment artifacts. This command supports testing workflows by ensuring clean state for MCP tool calls and service health verification.

## Command Interface

```bash
python . reset              # Default: clears RAG indices only
python . reset --all        # Full purge: indices + models + docker + temp
python . reset --index      # RAG indices only
python . reset --models     # Embedding models only
python . reset --docker     # Docker containers, volumes, images
python . reset --temp       # Test indices, caches
python . reset -i -d        # Combinations supported
```

### Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--all` | `-a` | Full environment reset |
| `--index` | `-i` | Clear RAG vector indices |
| `--models` | `-m` | Clear downloaded embedding models |
| `--docker` | `-d` | Purge Docker containers, volumes, images |
| `--temp` | `-t` | Clear test indices and temp files |

### Behavior

- No flags defaults to `--index`
- `--all` combines all flags
- No confirmation prompts (data is transient)
- Verbose output showing each action
- Exit code 0 on success, non-zero on failure
- Continues on partial failures, reports at end

## Artifact Breakdown

### `--index`: RAG Indices

Location: `.corpus/index/`

| File | Purpose |
|------|---------|
| `index.faiss` | FAISS binary index |
| `index.pkl` | Metadata pickle |
| `index.store.json` | Store metadata |

Also deletes: `data/test-index/`

### `--models`: Embedding Models

Location: `.corpus/models/`

Downloaded sentence-transformer models. Re-downloadable on next index build.

### `--docker`: Docker Stack

**Containers** (stop and remove):
- `wfmcp-server`
- `wfmcp-kroki`
- `wfmcp-kroki-d2`
- `wfmcp-kroki-mermaid`
- `wfmcp-kroki-bpmn`
- `wfmcp-kroki-excalidraw`

**Volumes** (remove):
- `wfmcp-corpus-data`
- `wfmcp-corpus-models`
- `wfmcp-output`

**Images** (remove):
- `wireframe-mcp-wfmcp-server` (custom built)
- Kroki images (yuzutech/kroki, renderers)

### `--temp`: Temporary Files

| Path | Purpose |
|------|---------|
| `data/test-index/` | Pytest test indices |
| `.pytest_cache/` | Pytest cache |
| `**/__pycache__/` | Python bytecode cache |

## Output Format

```
[reset] Clearing RAG indices...
  Deleted .corpus/index/index.faiss
  Deleted .corpus/index/index.pkl
  Deleted .corpus/index/index.store.json
  Deleted data/test-index/

[reset] Purging Docker stack...
  Stopped wfmcp-server
  Stopped wfmcp-kroki
  Removed container wfmcp-server
  Removed volume wfmcp-corpus-data
  Removed image wireframe-mcp-wfmcp-server:latest

[reset] Done. Environment reset complete.
```

## Implementation

### New Module: `src/reset/`

```
src/reset/
  __init__.py    # Exports
  lib.py         # Core cleanup functions
```

### Core Functions

```python
def clear_indices(verbose: bool = True) -> bool:
    """Delete RAG indices from .corpus/index/ and data/test-index/."""

def clear_models(verbose: bool = True) -> bool:
    """Delete embedding models from .corpus/models/."""

def clear_docker(verbose: bool = True) -> bool:
    """Stop containers, remove volumes, remove images."""

def clear_temp(verbose: bool = True) -> bool:
    """Delete pytest cache, pycache directories."""

def reset_environment(
    index: bool = False,
    models: bool = False,
    docker: bool = False,
    temp: bool = False,
    verbose: bool = True
) -> bool:
    """Orchestrate reset based on flags."""
```

### CLI Handler

Add to `__main__.py`:

```python
@cli.command()
@click.option("--all", "-a", "all_", is_flag=True, help="Full environment reset")
@click.option("--index", "-i", is_flag=True, help="Clear RAG indices")
@click.option("--models", "-m", is_flag=True, help="Clear embedding models")
@click.option("--docker", "-d", is_flag=True, help="Purge Docker stack")
@click.option("--temp", "-t", is_flag=True, help="Clear temp files")
def reset(all_, index, models, docker, temp):
    """Reset development environment artifacts."""
```

### Error Handling

- Each cleaner returns success boolean
- Missing files/directories are not errors
- Docker not running: skip docker cleanup, continue with others
- Report all failures at end
- Non-zero exit if any cleaner failed

## Task Phases

### Phase 1: Implement Reset Command
1. Create `src/reset/` module
2. Implement cleanup functions
3. Add CLI command with flags
4. Test each cleanup path

### Phase 2: Investigate index.models Bug
The vector store may be creating `index.models/` inside the index directory. Models should live in `.corpus/models/` as shared cross-index assets.

1. Audit `src/vector/lib.py` model storage logic
2. Fix to use `.corpus/models/` consistently
3. Separate PR if changes are invasive

### Phase 3: CLI Shorthand Audit
Ensure all CLI commands have consistent shorthand flags.

1. Audit existing commands
2. Document naming convention
3. Add missing shorthands
