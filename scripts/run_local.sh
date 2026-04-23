#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# run_local.sh — Start OmniGuard-Evolved-V2 locally in lightweight mode.
# Uses 2 env instances, no oracle bootstrap, no Redis requirement.
# -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export OMNIGUARD_ENV_INSTANCES="${OMNIGUARD_ENV_INSTANCES:-2}"
export OMNIGUARD_QUEUE_SIZE="${OMNIGUARD_QUEUE_SIZE:-100}"
export OMNIGUARD_MAX_LATENCY_STEPS="${OMNIGUARD_MAX_LATENCY_STEPS:-20}"
export OMNIGUARD_EPISODE_LENGTH="${OMNIGUARD_EPISODE_LENGTH:-8}"
export OMNIGUARD_DISABLE_ORACLE_BOOTSTRAP="${OMNIGUARD_DISABLE_ORACLE_BOOTSTRAP:-1}"
export OMNIGUARD_USE_TRANSFORMER_EMBEDDER="${OMNIGUARD_USE_TRANSFORMER_EMBEDDER:-0}"
export OMNIGUARD_HOST="${OMNIGUARD_HOST:-0.0.0.0}"
export OMNIGUARD_PORT="${OMNIGUARD_PORT:-8000}"

echo "[run_local] Project root: $PROJECT_ROOT"
echo "[run_local] Env instances: $OMNIGUARD_ENV_INSTANCES"
echo "[run_local] Oracle bootstrap: disabled"
echo "[run_local] Transformer embedder: disabled (hash fallback)"
echo "[run_local] Starting on http://${OMNIGUARD_HOST}:${OMNIGUARD_PORT} ..."

cd "$PROJECT_ROOT"
exec python -m uvicorn server.app:app \
    --host "$OMNIGUARD_HOST" \
    --port "$OMNIGUARD_PORT" \
    --log-level info
