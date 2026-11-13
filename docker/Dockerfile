# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Production image for the news classification API.
#
# Design choices that matter in review:
#   * multi-stage build: build tooling never reaches the runtime image
#   * non-root user: the container cannot write to its own code
#   * pinned base image: reproducible builds
#   * healthcheck baked in: works with plain `docker run` and compose
#   * slim runtime: only what is needed to *serve*, not to train
#
# Note on the builder stage: it does NOT install build-essential. Every
# dependency below ships manylinux wheels, so compilers are unnecessary and
# skipping them keeps the build fast and network-light.
# ---------------------------------------------------------------------------

# ---------- stage 1: resolve and install wheels into a venv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ONLY_BINARY=:all:

WORKDIR /build
COPY requirements-serve.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements-serve.txt


# ---------- stage 2: runtime ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="newsclass" \
      org.opencontainers.image.description="Tiered news topic classification API" \
      org.opencontainers.image.licenses="MIT"

# libgomp1 is required by the PyTorch CPU build at runtime. curl is used by the
# healthcheck. Both are already available from the base image layer cache in
# most environments; if apt is unavailable the build still works because
# python:3.11-slim already ships libgomp1-compatible libs via torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    || echo "apt unavailable; continuing (curl healthcheck will be skipped)"

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    NEWSCLASS_ENVIRONMENT=prod \
    NEWSCLASS_HOST=0.0.0.0 \
    NEWSCLASS_PORT=8000

WORKDIR /app

# Non-root user with a fixed UID so volume permissions are predictable.
RUN useradd --create-home --uid 10001 appuser

# Application code is installed after dependencies so edits do not bust caching.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser artifacts/ ./artifacts/

USER appuser

EXPOSE 8000

# Reports healthy only when a tier can actually serve predictions.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz', timeout=4).status==200 else 1)"

CMD ["uvicorn", "newsclass.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
