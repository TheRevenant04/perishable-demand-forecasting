# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: install runtime dependencies and the project into /app/.venv
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime stage: minimal image serving the trained model
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PERISHABLE_MODELS_DIR=/app/models \
    HOST=0.0.0.0 \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 1000 app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --chown=app:app models ./models
COPY --chown=app:app predict.py ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "predict.py"]
