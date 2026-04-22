from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True, frozen=True)
class CurriculumProfile:
    phase: str
    malicious_ratio: float
    obfuscation_probability: float
    semantic_noise: float
    stdio_escape_probability: float


class AdaptiveCurriculumScheduler:
    """Meta-learner that adjusts difficulty based on the agent's moving-average reward.

    Phase thresholds and minimum-episode guards prevent premature escalation.
    All thresholds are overridable via environment variables for experimentation.
    """

    def __init__(self, alpha: float = 0.08) -> None:
        self._alpha = alpha
        self._moving_reward = 0.0
        self._episodes_seen = 0
        self._lock = Lock()

        # Configurable phase boundaries
        self._phase_1_threshold = float(os.getenv("OMNIGUARD_PHASE1_THRESHOLD", "0.20"))
        self._phase_2_threshold = float(os.getenv("OMNIGUARD_PHASE2_THRESHOLD", "0.55"))
        self._min_episodes_per_phase = int(os.getenv("OMNIGUARD_MIN_EPISODES_PER_PHASE", "50"))

        # Track how many episodes have been spent in each phase for logging
        self._phase_episode_counts: dict[str, int] = {
            "bootstrapping": 0,
            "evasion_obfuscation": 0,
            "chained_exploitation": 0,
        }
        self._current_locked_phase: str | None = None
        self._phase_entry_episode = 0

    def update(self, episode_reward: float) -> None:
        with self._lock:
            if self._episodes_seen == 0:
                self._moving_reward = episode_reward
            else:
                self._moving_reward = (
                    self._alpha * episode_reward + (1.0 - self._alpha) * self._moving_reward
                )
            self._episodes_seen += 1
            current = self.current_phase()
            self._phase_episode_counts[current] = (
                self._phase_episode_counts.get(current, 0) + 1
            )

    def moving_reward(self) -> float:
        with self._lock:
            return self._moving_reward

    def episodes_seen(self) -> int:
        with self._lock:
            return self._episodes_seen

    def current_phase(self) -> str:
        score = self._moving_reward
        episodes = self._episodes_seen
        min_ep = self._min_episodes_per_phase

        # Determine candidate phase from reward signal
        if score < self._phase_1_threshold:
            candidate = "bootstrapping"
        elif score < self._phase_2_threshold:
            candidate = "evasion_obfuscation"
        else:
            candidate = "chained_exploitation"

        # Prevent premature advancement: require minimum episodes in current phase.
        if self._current_locked_phase is None:
            self._current_locked_phase = candidate
            self._phase_entry_episode = episodes

        if candidate != self._current_locked_phase:
            episodes_in_current = episodes - self._phase_entry_episode
            if episodes_in_current >= min_ep:
                self._current_locked_phase = candidate
                self._phase_entry_episode = episodes
        return self._current_locked_phase

    def phase_episode_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._phase_episode_counts)

    def profile(self) -> CurriculumProfile:
        phase = self.current_phase()
        if phase == "bootstrapping":
            return CurriculumProfile(
                phase=phase,
                malicious_ratio=0.35,
                obfuscation_probability=0.20,
                semantic_noise=0.10,
                stdio_escape_probability=0.08,
            )
        if phase == "evasion_obfuscation":
            return CurriculumProfile(
                phase=phase,
                malicious_ratio=0.55,
                obfuscation_probability=0.45,
                semantic_noise=0.35,
                stdio_escape_probability=0.20,
            )
        return CurriculumProfile(
            phase=phase,
            malicious_ratio=0.72,
            obfuscation_probability=0.70,
            semantic_noise=0.60,
            stdio_escape_probability=0.36,
        )
