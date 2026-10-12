# Multi-stage production Dockerfile for High-Load FastAPI Enterprise Gateway
FROM python:3.12-slim AS runtime

# System environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime utilities for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20240203 \
    curl=7.88.1-10+deb12u8 \
    && rm -rf /var/lib/apt/lists/*

# Security: Dedicated non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app -M appuser

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# Copy application source code
COPY app/ /app/app/

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to unprivileged user
USER appuser

# Expose standard application port
EXPOSE 8000

# Docker healthcheck pinging internal FastAPI health endpoint
HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server launch command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--access-log"]
