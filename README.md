---
title: OmniGuard Evolved V2
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv RL environment for training MCP gateway defenders.
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



# ⚔️ OmniGuard-Evolved-V2

> _The attacker is an AI. It moves at machine speed. It never gets tired. It learns from every block. So we built a defender that does too._

A distributed adversarial RL environment that trains a language model to defend MCP gateways against autonomous AI attacks.

```
MYTHOS-CLASS ATTACKER
[mutate] ──► [re-inject] ──► [sandbox escape] ──► [exfiltrate]
                                    │
                         ── MCP GATEWAY ──
                                    │
OMNIGUARD DEFENDER
[observe] ──► [classify] ──► [decide] ──► [learn] ──► [adapt]
```

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Built with Unsloth](https://img.shields.io/badge/training-Unsloth%20%2B%20GRPO-orange.svg)](https://github.com/unslothai/unsloth)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-OmniGuard--Evolved--V2-yellow.svg)](https://huggingface.co/spaces/SmartKapila/OmniGuard-Evolved-V2)
[![WandB](https://img.shields.io/badge/WandB-Training%20Run-orange.svg)](https://wandb.ai/smartatk04-thapar-university/omniguard-vulnops-v3)

---

## Links

| Resource              | URL                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 🤗 HuggingFace Space  | [https://huggingface.co/spaces/SmartKapila/OmniGuard-Evolved-V2](https://huggingface.co/spaces/SmartKapila/OmniGuard-Evolved-V2)                     |
| 📊 WandB Training Run | [Weights and Biases Dashboard for training run](https://wandb.ai/smartatk04-thapar-university/omniguard-vulnops-v3) |
| 📝 Blog Post          | [Read the full writeup](https://huggingface.co/spaces/SmartKapila/OmniGuard-Evolved-V2/blob/main/BLOG.md)                       |
| 🎓 Colab Notebook     | [Re-run training yourself](https://colab.research.google.com/drive/1hVkLiQBlHXCTzca3UK2W78dsUJfMDk4D?usp=sharing)               |

---

## Threat Model

Modern AI assistants operate through MCP, giving language models structured access to tools — file systems, APIs, shell commands. Adversarial AI exploits this via prompt injection, sandbox escapes, and credential exfiltration at machine speed. The **Action Calibration Gap** is the core challenge: blocking everything causes business downtime; missing an attack breaches the network. OmniGuard trains a defender to navigate this gap.

---

## Three Anti-Mythos Mechanics

### ⚡ USP 1: Recursive Self-Correction Trap

When the defender blocks an attack, there's a **35% chance the next payload is a mutated variant** via the `PayloadMutator` — using base64 encoding, URL encoding, or semantic synonym substitution. The defender must maintain elevated vigilance after every block.

### ⏱️ USP 2: Sub-20 Step Temporal Decay

Each episode has a **20-step latency budget**. Overuse of deferred review incurs up to **−0.35** penalty, shaping the policy toward machine-speed triage.

### 🔓 USP 3: STDIO Sandbox Escape

Payloads with STDIO attack markers (`stdio`, `fork bomb`, `tty hijack`, etc.) require `REVOKE_STDIO` — a generic `BLOCK` is scored as a **false negative (−1.0, episode terminates)**. This forces a genuinely differentiated policy.

So in this way we are creating an RL training environment that mimics how **Claude Mythos** initiates MCP level agentic attacks. 
Thanks to this [Report](https://drive.google.com/file/d/1U-gCVXpYpcA_O1GzDuQ-F-oILhScCxIC/view?usp=sharing) which helped us study and analyse how Claude inittiates its attack pipeline by mutating payloads, creating long level MCP tool call chains etc.
---

## What Makes This Different

| Feature           | Standard Env  | OmniGuard                                        |
| ----------------- | ------------- | ------------------------------------------------ |
| Adversary         | Fixed dataset | Adaptive — mutates after every block             |
| Temporal pressure | None          | 20-step latency budget with decay                |
| Attack taxonomy   | Generic       | STDIO escapes requiring channel revocation       |
| Data source       | Static        | Live HuggingFace streaming, 100M+ samples        |
| Difficulty        | Fixed         | 3-phase curriculum, auto-advances on performance |
| Reward            | Single signal | 4 independent composable sub-functions           |
| Anti-cheat        | None          | Verifier catches reward hacking in rationale     |
| Scale             | Single env    | 32 parallel process-isolated instances           |
| Multi-agent       | No            | Defender + Adversary + SemanticCritic Judge      |

---

## Agents

- **🛡️ Defender — Qwen2.5-3B:** Reads every MCP request and chooses one of six defensive actions. Trained via GRPO across all four reward signals simultaneously.
- **🗡️ Adversary — PayloadMutator + AdaptiveCurriculum:** Procedural attacker that escalates difficulty and spawns mutated variants after each successful block. No learned weights — pure environmental mechanics.
- **⚖️ Critic — SemanticCritic + DeterministicOracle:** Oracle uses a 50,000-entry NIST dataset for authoritative labelling. SemanticCritic activates on `SEMANTIC_DIFF` actions, computing embedding similarity to reward recursive attack identification (0.1–0.8).

---

## The Six Actions

| Action                 | When to use                 | Anti-Mythos role                                 |
| ---------------------- | --------------------------- | ------------------------------------------------ |
| `ALLOW`                | Clearly benign traffic      | FN terminates episode                            |
| `BLOCK`                | General malicious traffic   | Insufficient for STDIO attacks                   |
| `SPOTLIGHT`            | Uncertain — flag for review | Safe default, but overuse incurs latency penalty |
| `SEMANTIC_DIFF`        | Obfuscated/mutated payloads | Activates SemanticCritic for recursive detection |
| `CAPABILITY_MEDIATION` | MCP privilege abuse         | Must include restrictions or gets penalized      |
| `REVOKE_STDIO`         | STDIO sandbox escapes       | Only correct response to channel hijacking       |

---

## Reward Design

Four independent sub-functions compose the total reward:

**1. Security Signal**

```
True Positive  (caught real attack)         →  +0.50
True Negative  (correctly allowed benign)   →  +0.20
False Positive (blocked legitimate traffic) →  −0.40
False Negative (missed attack)              →  −1.00  + EPISODE TERMINATED
```

_STDIO rule: `BLOCK` on a sandbox escape scores as False Negative._

**2. Usability Signal** — penalizes repeated ALLOW on malicious traffic (−0.30), prompt injection in rationale (−0.15), `CAPABILITY_MEDIATION` with no restrictions (−0.08).

**3. Latency Signal** — linear decay up to −0.35 over the 20-step budget.

**4. Format Signal** — +0.02 for meaningful rationale, −0.05 for empty output.

Episode scores are normalized via **micro-F1** and drive curriculum advancement.

---

## Curriculum

```
Phase 1: Bootstrapping       Malicious: 35% │ Obfuscation: 20% │ STDIO:  8%
Phase 2: Evasion             Malicious: 55% │ Obfuscation: 45% │ STDIO: 20%
Phase 3: Chained Exploitation Malicious: 72% │ Obfuscation: 70% │ STDIO: 36%
```

Advancement uses exponential moving average (α=0.08). Thresholds: >0.20 for Phase 2, >0.55 for Phase 3 (min 50 episodes each).

---

## Anti-Reward-Hacking Verifier

Runs before any reward is computed. Flags include: `contradictory_allow`, `monotonic_action` (last 8 identical), `reward_hacking_detected`, `rationale_suspicious`, `missing_restrictions`, `repeated_allow_risk`.

---

## WandB Results

> Full run: [wandb.ai/smartatk04-thapar-university/omniguard-vulnops-v3](https://wandb.ai/smartatk04-thapar-university/omniguard-vulnops-v3?nw=nwusersmartatk04) · 500 steps · Qwen2.5-3B · GRPO

### Key Metrics

| Metric                      | Early (steps 1–130)      | Mid (steps 130–220)         | Late (steps 220–500)           | Final value |
| --------------------------- | ------------------------ | --------------------------- | ------------------------------ | ----------- |
| **Loss**                    | −0.20 → +0.20 (volatile) | Moderate spikes             | ≈ 0, occasional spikes ≤ +0.20 | ≈ 0.00      |
| **Reward (mean)**           | −4.0 → +4.0 (unstable)   | Transition, dips to −0.5    | Converges +2.0 → +3.0          | ≈ +2.5      |
| **Reward std**              | 1.0 – 1.8                | Peak spike ≈ 3.0            | Settles 1.0 – 1.5              | ≈ 1.2       |
| **KL divergence**           | ≈ 0.02 – 0.05            | Spike to ≈ 1.0 at step ~135 | Steady 0.10 – 0.30             | ≈ 0.15      |
| **Gradient norm**           | 0.40 – 0.80              | Peak ≈ 1.4 at step ~135     | 0.20 – 0.90                    | ≈ 0.55      |
| **Learning rate**           | 0 → 5e−6 (warm-up)       | Cosine decay begins         | Decays smoothly → 0            | ≈ 0         |
| **Env step reward (mean)**  | −3.0 → +3.0 (unstable)   | Dips to −1.0                | +1.5 → +2.5                    | ≈ +2.0      |
| **Env step reward (std)**   | ≈ 3.0 (high)             | Drops significantly         | 0.50 – 1.50                    | ≈ 0.90      |
| **Threat awareness (mean)** | −1.0 → +1.0 (volatile)   | Rapid improvement           | +0.80 → +1.0                   | ≈ +0.95     |
| **Threat awareness (std)**  | ≈ 1.0 (max)              | Sharp drop post step ~200   | 0.0 – 0.60                     | ≈ 0.40      |

> **Note:** Threat awareness converges to ≈ +0.95 by step ~220 — the clearest signal of a learned defensive policy. The KL spike at step ~135 marks a large distributional shift that resolved within ~80 steps. Total reward stabilises in the +2 → +3 band for the remainder of training.

---

## Quick Start

```bash
git clone <your-repo-url> && cd OmniGuard-Evolved-V2
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
bash scripts/run_local.sh
```

**Docker:** `docker compose up --build` (starts Redis → Data Worker → Environment API)

**Training:**

```bash
accelerate launch --config_file config/accelerate_fsdp.yaml \
    training/grpo_distributed.py --env-url http://127.0.0.1:8000 --project omniguard-openenv
```

**Benchmark:**

```bash
python -m eval.benchmark --env-url http://127.0.0.1:8000 --steps 1000 \
    --trained-adapter-path outputs/grpo-distributed --output-dir reports
```

---

## API Reference

| Method | Path       | Description                       |
| ------ | ---------- | --------------------------------- |
| `GET`  | `/healthz` | Liveness probe                    |
| `GET`  | `/readyz`  | Readiness probe                   |
| `GET`  | `/info`    | Full environment spec             |
| `POST` | `/reset`   | Reset environment instances       |
| `POST` | `/step`    | Submit batch of defensive actions |
| `GET`  | `/metrics` | Per-instance telemetry            |

---

## Key Environment Variables

| Variable                             | Default | Description                      |
| ------------------------------------ | ------- | -------------------------------- |
| `OMNIGUARD_ENV_INSTANCES`            | `32`    | Parallel environment instances   |
| `OMNIGUARD_EPISODE_LENGTH`           | `16`    | Steps per episode                |
| `OMNIGUARD_MAX_LATENCY_STEPS`        | `20`    | Latency budget                   |
| `OMNIGUARD_QUEUE_SIZE`               | `1000`  | Payload prefetch queue depth     |
| `OMNIGUARD_DISABLE_ORACLE_BOOTSTRAP` | `0`     | Skip NIST hydration (fast start) |
| `OMNIGUARD_USE_TRANSFORMER_EMBEDDER` | `1`     | Sentence-transformer embeddings  |
| `OMNIGUARD_REDIS_URL`                | —       | Redis for telemetry              |
| `OMNIGUARD_PHASE1_THRESHOLD`         | `0.20`  | Score to advance to Phase 2      |
| `OMNIGUARD_PHASE2_THRESHOLD`         | `0.55`  | Score to advance to Phase 3      |
| `OMNIGUARD_MIN_EPISODES_PER_PHASE`   | `50`    | Min episodes before advance      |

---

## Tech Stack

Python 3.12 · FastAPI + uvloop · Pydantic V2 · HuggingFace datasets (streaming) · TRL GRPO · Unsloth 4-bit LoRA · Accelerate + FSDP (8× GPU) · WandB · Redis · Docker Compose · sentence-transformers/all-MiniLM-L6-v2

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

_Built by Team Epochalypse for the OpenEnv Hackathon._
