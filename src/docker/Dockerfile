# syntax=docker/dockerfile:1

# ============================================================================
# wireframe-mcp Docker Image
# Multi-stage build for the MCP server with CUDA 12+ and Python 3.12 support
# ============================================================================

# ----------------------------------------------------------------------------
# Base stage: CUDA 12.4.1 Runtime Ubuntu 22.04
# ----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

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

# Install Python dependencies (including faiss-gpu)
# Note: faiss-gpu wheels often ship with their own CUDA libs, but we have system CUDA here too.
# We trust pip to resolve the compatible wheel.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# ----------------------------------------------------------------------------
# Runtime stage: Final image
# ----------------------------------------------------------------------------
FROM base AS runtime

# Create non-root user for security (although accessing /dev/nvidia might require specific groups in some runtimes)
RUN groupadd --gid 1000 mcp \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home mcp

# Copy installed packages from builder
# We find where site-packages are located for python 3.12
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create directories for corpus data and config
RUN mkdir -p /app/corpus /app/config \
    && chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Volume mount points
VOLUME ["/app/corpus", "/app/config"]

# Environment variables for MCP configuration
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8080 \
    CORPUS_PATH=/app/corpus \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose MCP server port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src import list_providers; list_providers()" || exit 1

# Default entrypoint
ENTRYPOINT ["python", "-m", "src"]
