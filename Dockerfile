FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data/evaluation ./data/evaluation

RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    DATA_DIR=/app/data \
    RAG_CORPUS_DIR=/app/data/chunks \
    THESIS_ARTIFACT_BASE_DIR=/app/data/reports

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system app \
    && adduser --system --ingroup app --home /app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml ./pyproject.toml
COPY --from=builder /app/README.md ./README.md
COPY --from=builder /app/data/evaluation ./data/evaluation

RUN mkdir -p /app/data/chunks /app/data/normalized /app/data/raw /app/data/reports \
    && chown -R app:app /app /opt/venv

USER app

EXPOSE 8000

CMD ["stock-agent-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
