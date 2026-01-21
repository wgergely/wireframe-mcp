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
