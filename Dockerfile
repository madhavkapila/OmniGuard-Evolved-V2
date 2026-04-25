# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# OmniGuard-Evolved-V2 -- HuggingFace Space deployment image
# Builds a slim CPU-only container (~600 MB) that runs the FastAPI/OpenEnv
# server. Training-only dependencies (torch, unsloth, trl, peft, ...) are
# NOT installed here; they belong in the training notebook only.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf-cache \
    TRANSFORMERS_CACHE=/tmp/hf-cache \
    HF_DATASETS_CACHE=/tmp/hf-cache \
    OMNIGUARD_HOST=0.0.0.0 \
    OMNIGUARD_PORT=7860 \
    # ---- HF Spaces memory-safe defaults (override in Space "Variables") ----
    OMNIGUARD_ENV_INSTANCES=2 \
    OMNIGUARD_QUEUE_SIZE=128 \
    OMNIGUARD_EPISODE_LENGTH=8 \
    OMNIGUARD_MAX_LATENCY_STEPS=12 \
    OMNIGUARD_UVICORN_WORKERS=1 \
    OMNIGUARD_WORKER_TIMEOUT_SEC=120 \
    OMNIGUARD_USE_TRANSFORMER_EMBEDDER=0 \
    OMNIGUARD_DISABLE_ORACLE_BOOTSTRAP=1 \
    OMNIGUARD_ORACLE_CACHE=2000 \
    OMNIGUARD_LOG_LEVEL=info \
    OMNIGUARD_MAX_REQUEST_BYTES=4000000 \
    OMNIGUARD_BASE_SEED=3407

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir --no-deps openenv-core>=0.2.3

COPY server      /app/server
COPY config      /app/config
COPY openenv.yaml /app/openenv.yaml
COPY README.md   /app/README.md

# HF Spaces runs as UID 1000 by default; create a matching user and a
# writable cache dir so transformers/datasets don't try to mkdir in /.
RUN groupadd -r omniguard && useradd -r -g omniguard -u 1000 -d /app omniguard \
    && mkdir -p /tmp/hf-cache \
    && chown -R omniguard:omniguard /app /tmp/hf-cache
USER omniguard

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:${OMNIGUARD_PORT}/healthz || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host ${OMNIGUARD_HOST} --port ${OMNIGUARD_PORT} --loop uvloop --http httptools --workers ${OMNIGUARD_UVICORN_WORKERS}"]
