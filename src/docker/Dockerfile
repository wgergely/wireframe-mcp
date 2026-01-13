# syntax=docker/dockerfile:1

# ============================================================================
# wireframe-mcp Docker Image
# Multi-stage build for the MCP server with corpus data support
# ============================================================================

# ----------------------------------------------------------------------------
# Base stage: Python 3.12 slim
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ----------------------------------------------------------------------------
# Builder stage: Install dependencies
# ----------------------------------------------------------------------------
FROM base AS builder

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# ----------------------------------------------------------------------------
# Runtime stage: Final image
# ----------------------------------------------------------------------------
FROM base AS runtime

# Create non-root user for security
RUN groupadd --gid 1000 mcp \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home mcp

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
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
    CORPUS_PATH=/app/corpus

# Expose MCP server port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src import list_providers; list_providers()" || exit 1

# Default entrypoint (placeholder - update when MCP server is implemented)
ENTRYPOINT ["python", "-m", "src"]
