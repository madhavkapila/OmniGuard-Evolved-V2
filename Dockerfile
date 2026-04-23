FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r omniguard && useradd -r -g omniguard -d /app omniguard

COPY pyproject.toml README.md requirements.txt /app/
COPY server /app/server
COPY training /app/training
COPY eval /app/eval
COPY worker /app/worker
COPY config /app/config

RUN pip install --upgrade pip && pip install -e .

RUN chown -R omniguard:omniguard /app
USER omniguard

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
