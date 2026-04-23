---
title: OmniGuard Evolved V2
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: A distributed OpenEnv RL environment for training LLM-based defenders to protect enterprise MCP gateways against autonomous adversarial AI attacks.
tags:
  - openenv
  - reinforcement-learning
  - ai-security
  - mcp
  - fastapi
  - pytorch
  - Unsloth
  - Hugging Face
---


# OmniGuard-Evolved-V2

**Distributed OpenEnv RL Environment for Autonomous VulnOps & MCP Gateway Defense**

> An asymmetric multi-agent reinforcement learning environment that trains an LLM-based
> defender to protect enterprise MCP (Model Context Protocol) gateways against autonomous
> adversarial AI attacks — including prompt injection, credential exfiltration, STDIO
> sandbox escapes, and recursive self-correction chains.

### 🏆 Hackathon Submission Links
- **Hugging Face Space**: [OmniGuard-Evolved-V2 Environment](https://huggingface.co/spaces/omni-team/omniguard-evolved-v2) *(Replace with actual URL before submission)*
- **2-Minute Pitch Video**: [YouTube Link](https://youtube.com) *(Replace with actual URL before submission)*

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Training Cluster (Accelerate + FSDP)                   │
│  ┌──────────────────────────────────┐                   │
│  │  grpo_distributed.py            │                    │
│  │  Qwen2.5-3B + LoRA (Unsloth)   │◄── WandB Logging  │
│  └──────────┬───────────────────────┘                   │
│             │ HTTP: /step, /reset                       │
├─────────────┼───────────────────────────────────────────┤
│  Environment API (FastAPI + uvloop)                     │
│  ┌──────────▼──────────┐  ┌────────────────────┐       │
│  │  AsyncVectorEnvMgr  │  │  StreamingPayload  │       │
│  │  32× ProcessPool    │  │  Generator         │       │
│  │  workers             │  │  (HF datasets      │       │
│  │  ┌────────────────┐ │  │   streaming=True)  │       │
│  │  │ StateMachine   │ │  └────────────────────┘       │
│  │  │ ├─ Verifier    │ │                               │
│  │  │ ├─ DualGrader  │ │  ┌────────────────────┐       │
│  │  │ ├─ Curriculum  │ │  │  Redis (telemetry) │       │
│  │  │ └─ Telemetry   │ │  └────────────────────┘       │
│  │  └────────────────┘ │                               │
│  └─────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

## Multi-Agent Dynamics

| Agent | Role | Goal |
|---|---|---|
| **Defender** (Qwen2.5 RL) | Primary — your trained model | Maximise reward by classifying traffic accurately |
| **Adversary** (PayloadMutator + Curriculum) | Procedural attacker | Escalate obfuscation as defender improves |
| **Critic** (SemanticCritic) | LLM-as-a-Judge | Grade semantic-diff actions via embedding similarity |

## Reward Design (Independent Sub-Functions)

The grader composes **four independent reward signals**, per organiser guidance:

1. **Security** — TP (+0.5), TN (+0.2), FP (−0.4), FN (−1.0 + episode termination)
2. **Usability** — Penalises repeated-allow, suspicious rationale, missing MCP restrictions
3. **Latency** — Temporal decay when analysis exceeds budget
4. **Format** — Rewards well-formed JSON rationale, penalises empty/lazy output

## Quick Start

```bash
# 1. Clone and enter repo
git clone <your-repo-url> && cd OmniGuard-Evolved-V2

# 2. Create virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Copy and edit environment config
cp .env.example .env

# 5. Start (lightweight local mode)
OMNIGUARD_ENV_INSTANCES=2 OMNIGUARD_DISABLE_ORACLE_BOOTSTRAP=1 \
    uvicorn server.app:app --host 0.0.0.0 --port 8000

# 6. Verify
curl http://localhost:8000/healthz
curl http://localhost:8000/info
```

### Docker Compose (Full Stack)

```bash
docker compose up --build
```

This starts: Redis → Data Worker → Environment API (32 instances)

### Training

```bash
accelerate launch --config_file config/accelerate_fsdp.yaml \
    training/grpo_distributed.py \
    --env-url http://127.0.0.1:8000 \
    --project omniguard-openenv
```

### Benchmarking

```bash
python -m eval.benchmark \
    --env-url http://127.0.0.1:8000 \
    --steps 1000 \
    --output-dir reports
```

Produces `reports/results.json` and `reports/reward_curve.png`.

#### Empirical Improvement Proof

The graphs below demonstrate the empirical improvement of the GRPO-trained policy over the untrained baseline, showing both the increase in overall reward and the massive reduction in "Alert Fatigue" (False Positive rate).

![Reward and False Positive Curves](reports/reward_curve.png)

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe |
| GET | `/info` | Environment specification (action/observation space) |
| GET | `/metadata` | Runtime metadata |
| POST | `/reset` | Reset environment instances |
| POST | `/step` | Execute batch of defense actions |
| GET | `/metrics` | Per-instance telemetry |

## Tech Stack

- **Python 3.12** + **FastAPI** (async, uvloop, httptools)
- **Pydantic V2** strict contracts
- **Hugging Face datasets** (streaming mode)
- **TRL** (GRPO trainer) + **Unsloth** (4-bit LoRA, gradient checkpointing)
- **Accelerate + FSDP** (multi-GPU distribution)
- **WandB** (experiment tracking)
- **Redis** (cross-environment telemetry)
- **Docker Compose** (production deployment)

## Project Structure

```
OmniGuard-Evolved-V2/
├── server/
│   ├── app.py              # FastAPI application + lifespan
│   ├── models.py           # Pydantic V2 schemas
│   ├── vector_env.py       # Multiprocess vectorized env manager
│   ├── env.py              # Per-instance state machine
│   ├── generator.py        # Streaming payload generator + mutator
│   ├── graders.py          # Dual-grader: Oracle + SemanticCritic
│   ├── verifier.py         # Anti-reward-hacking action verifier
│   ├── curriculum.py       # Adaptive difficulty scheduler
│   ├── embeddings.py       # Dynamic embedding generation
│   ├── telemetry.py        # Redis-backed telemetry sink
│   ├── payloads.py         # Data contracts + constants
│   └── openenv_adapter.py  # OpenEnv compatibility layer
├── training/
│   └── grpo_distributed.py # GRPO training pipeline
├── eval/
│   └── benchmark.py        # Baseline vs trained policy benchmarks
├── worker/
│   └── data_worker.py      # Redis streaming data ingestor
├── config/
│   └── accelerate_fsdp.yaml
├── scripts/
│   ├── run_local.sh        # One-command local startup
│   └── verify_runtime.py   # Pre-flight dependency checker
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

## License

Apache-2.0 license

