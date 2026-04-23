from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EnvTelemetryState:
    rewards: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    false_positives: int = 0
    catastrophic_breaches: int = 0
    total_steps: int = 0


class TelemetrySink:
    def __init__(self, env_id: int, redis_url: str | None = None) -> None:
        self.env_id = env_id
        self.redis_url = redis_url or os.getenv("OMNIGUARD_REDIS_URL")
        self.state = EnvTelemetryState()
        self._redis = None
        if self.redis_url:
            try:
                import redis

                self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
            except Exception:
                self._redis = None

    def record_step(self, reward_total: float, info: dict[str, Any], phase: str) -> None:
        self.state.rewards.append(reward_total)
        self.state.total_steps += 1
        if info.get("false_positive"):
            self.state.false_positives += 1
        if info.get("catastrophic_breach"):
            self.state.catastrophic_breaches += 1

        if self._redis is not None:
            key = f"omniguard:env:{self.env_id}:telemetry"
            payload = {
                "reward_total": reward_total,
                "phase": phase,
                "false_positive": int(bool(info.get("false_positive"))),
                "catastrophic_breach": int(bool(info.get("catastrophic_breach"))),
                "timestamp": time.time(),
            }
            try:
                self._redis.xadd(key, {"data": json.dumps(payload)}, maxlen=1000, approximate=True)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("Telemetry error: %s", e)

    def snapshot(self) -> dict[str, Any]:
        rewards = list(self.state.rewards)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return {
            "env_id": self.env_id,
            "avg_reward": avg_reward,
            "false_positives": self.state.false_positives,
            "catastrophic_breaches": self.state.catastrophic_breaches,
            "total_steps": self.state.total_steps,
        }
