#!/usr/bin/env bash
# ================================================================
#  uv_commands.sh — Exact UV terminal commands for OmniGuard-Evolved-V2
#  Matching the mentors' execution style from the Opening Ceremony deck.
# ================================================================

set -euo pipefail

# -----------------------------------------------------------------
#  1. Install UV (if not already available)
# -----------------------------------------------------------------
# pip install --upgrade uv
# or:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# -----------------------------------------------------------------
#  2. Create virtual environment and install the environment
# -----------------------------------------------------------------
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the project with all dependencies
uv pip install -e ".[openenv]"

# -----------------------------------------------------------------
#  3. Run the OpenEnv environment server (Mentor-style: local dev)
# -----------------------------------------------------------------
# Lightweight mode: 2 env instances, no oracle bootstrap, no Redis
OMNIGUARD_ENV_INSTANCES=2 \
OMNIGUARD_DISABLE_ORACLE_BOOTSTRAP=1 \
OMNIGUARD_USE_TRANSFORMER_EMBEDDER=0 \
uv run uvicorn server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info

# -----------------------------------------------------------------
#  4. Verify the environment is running
# -----------------------------------------------------------------
# curl http://localhost:8000/healthz
# curl http://localhost:8000/info

# -----------------------------------------------------------------
#  5. Run via Docker (Production deployment)
# -----------------------------------------------------------------
# docker compose up --build

# -----------------------------------------------------------------
#  6. Deploy to Hugging Face Spaces
# -----------------------------------------------------------------
# huggingface-cli repo create omniguard-evolved-v2 --type space --space-sdk docker
# git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/omniguard-evolved-v2
# git push hf main
