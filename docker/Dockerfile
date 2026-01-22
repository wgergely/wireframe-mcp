# syntax=docker/dockerfile:1

# ============================================================================
# wireframe-mcp Docker Image
# Multi-stage build with dev/prod targets for CUDA 12+ and Python 3.12
# ============================================================================

# ----------------------------------------------------------------------------
# Base stage: CUDA 12.6.3 Runtime Ubuntu 22.04
# ----------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS base

# Set environment variables for non-interactive installs and Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHON_VERSION=3.12

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` and `pip` point to Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

WORKDIR /app

# ----------------------------------------------------------------------------
# Builder stage: Install dependencies
# ----------------------------------------------------------------------------
FROM base AS builder

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install Python dependencies with GPU support
# CRITICAL: PyTorch MUST be installed with CUDA from PyTorch's index BEFORE sentence-transformers
# This ensures GPU acceleration is available for embeddings and model inference
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir faiss-gpu-cu12 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# ----------------------------------------------------------------------------
# Development target: Hot reloading support
# Source code mounted as volume, not copied into image
# ----------------------------------------------------------------------------
FROM base AS dev

# Create non-root user
RUN groupadd --gid 1000 mcp \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home mcp

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only pyproject.toml (source mounted at runtime)
COPY pyproject.toml ./

# Create directories for runtime data
RUN mkdir -p /app/corpus/data /app/corpus/models /app/config /app/data /app/src \
    && chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Volume mount points for persistent data
VOLUME ["/app/corpus", "/app/config", "/app/data"]

# Environment variables (aligned with src/config defaults)
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=18080 \
    KROKI_PORT=18000 \
    CORPUS_PATH=/app/corpus \
    CORPUS_DATA_DIR=/app/corpus/data \
    CORPUS_MODELS_DIR=/app/corpus/models \
    CORPUS_INDEX_DIR=/app/corpus/index \
    EMBEDDING_BACKEND=local \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 18080

# Dev mode: run with python -u for unbuffered output
# The project uses __main__.py at root (run with `python .`)
ENTRYPOINT ["python", "-u"]
CMD ["."]

# ----------------------------------------------------------------------------
# Production target: Optimized runtime image
# Source code copied into image for deployment
# ----------------------------------------------------------------------------
FROM base AS prod

# Create non-root user
RUN groupadd --gid 1000 mcp \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home mcp

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY docker/ ./docker/
COPY __main__.py pyproject.toml ./

# Create directories for runtime data
RUN mkdir -p /app/corpus/data /app/corpus/models /app/config /app/data \
    && chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Volume mount points for persistent data
VOLUME ["/app/corpus", "/app/config", "/app/data"]

# Environment variables (aligned with src/config defaults)
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=18080 \
    KROKI_PORT=18000 \
    CORPUS_PATH=/app/corpus \
    CORPUS_DATA_DIR=/app/corpus/data \
    CORPUS_MODELS_DIR=/app/corpus/models \
    CORPUS_INDEX_DIR=/app/corpus/index \
    EMBEDDING_BACKEND=local \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 18080

# Healthcheck validates core module imports
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src import list_providers; list_providers()" || exit 1

ENTRYPOINT ["python"]
CMD ["."]
