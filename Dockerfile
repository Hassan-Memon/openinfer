# syntax=docker/dockerfile:1.7
#
# Multi-stage build, CPU only. Stage 1 resolves and installs dependencies with uv into a
# virtualenv; stage 2 copies just that virtualenv onto a slim runtime image so the final
# image carries no compilers, no uv and no build cache.
# See docs/learn/03-concepts/ (Docker multi-stage is explained in 01-architecture-walkthrough.md).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first so this layer is cached while source files change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra metrics

# Now the package itself.
COPY openinfer ./openinfer
COPY README.md LICENSE openinfer.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra metrics


FROM python:3.12-slim-bookworm AS runtime

# Non-root user; Hugging Face Spaces also requires uid 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/openinfer /app/openinfer
COPY --from=builder --chown=app:app /app/openinfer.yaml /app/openinfer.yaml

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OPENINFER_HOST=0.0.0.0 \
    OPENINFER_PORT=8000 \
    OPENINFER_LEDGER_URL=sqlite:////data/openinfer.db

RUN mkdir -p /data && chown app:app /data
VOLUME ["/data"]
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"OPENINFER_PORT\",\"8000\")}/health').read()" || exit 1

CMD ["openinfer", "serve"]
