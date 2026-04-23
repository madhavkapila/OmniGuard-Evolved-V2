from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from datasets import load_dataset

from server.models import DefenseAction, DefenseActionType, StepReward
from server.payloads import BLOCKING_ACTIONS, MALICIOUS_KEYWORDS, ORACLE_DATASET_ID, PayloadSample


class DeterministicOracle:
    def __init__(self, cache_size: int = 50000) -> None:
        self.cache_size = cache_size
        self.lookup: dict[str, bool] = {}
        self._hydrate_oracle()

    def _fingerprint(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.lower()).strip()
        return cleaned

    def _extract_text(self, row: dict[str, Any]) -> str:
        for key in ("text", "content", "instruction", "prompt", "payload"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in row.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_label(self, row: dict[str, Any], text: str) -> bool:
        label_keys = ("label", "is_malicious", "class", "category", "type")
        for key in label_keys:
            value = row.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value > 0
            if isinstance(value, str):
                lowered = value.lower()
                if any(token in lowered for token in ("malicious", "attack", "threat", "exploit")):
                    return True
                if any(token in lowered for token in ("benign", "normal", "safe")):
                    return False
        lowered_text = text.lower()
        return any(keyword in lowered_text for keyword in MALICIOUS_KEYWORDS)

    def _hydrate_oracle(self) -> None:
        if os.getenv("OMNIGUARD_DISABLE_ORACLE_BOOTSTRAP", "0") == "1":
            self.lookup = {}
            return
        try:
            stream = load_dataset(ORACLE_DATASET_ID, split="train", streaming=True)
            count = 0
            for row in stream:
                text = self._extract_text(row)
                if not text:
                    continue
                label = self._extract_label(row, text)
                self.lookup[self._fingerprint(text)] = label
                count += 1
                if count >= self.cache_size:
                    break
        except Exception:
            # Operational via keyword heuristics when remote bootstrap fails.
            self.lookup = {}

    def is_malicious(self, text: str) -> bool:
        fp = self._fingerprint(text)
        if fp in self.lookup:
            return self.lookup[fp]
        lowered = text.lower()
        return any(keyword in lowered for keyword in MALICIOUS_KEYWORDS)


class SemanticCritic:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-critic")
        self._model = None
        self._tokenizer = None
        self._load_attempted = False

    def _load_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        except Exception:
            self._tokenizer = None
            self._model = None

    def _lexical_similarity(self, a: str, b: str) -> float:
        tokens_a = set(re.findall(r"[a-zA-Z0-9_]+", a.lower()))
        tokens_b = set(re.findall(r"[a-zA-Z0-9_]+", b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _model_similarity(self, a: str, b: str) -> float:
        self._load_model()
        if self._tokenizer is None or self._model is None:
            return 0.0
        try:
            import torch

            enc = self._tokenizer(
                [a, b],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            with torch.no_grad():
                out = self._model(**enc)
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            denom = np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
            if denom <= 0:
                return 0.0
            return float(np.dot(emb[0], emb[1]) / denom)
        except Exception:
            return 0.0

    def _score_sync(self, canonical: str, obfuscated: str) -> float:
        lexical = self._lexical_similarity(canonical, obfuscated)
        model = self._model_similarity(canonical, obfuscated)
        similarity = max(lexical, model)
        return float(np.clip(0.1 + (0.7 * similarity), 0.1, 0.8))

    async def score(self, canonical: str, obfuscated: str) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._score_sync, canonical, obfuscated)

    def score_sync(self, canonical: str, obfuscated: str) -> float:
        return self._score_sync(canonical, obfuscated)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Independent reward sub-functions
# Organizer guidance: "use multiple independent reward functions, not just one"
# ---------------------------------------------------------------------------

def compute_security_reward(
    action: DefenseAction,
    payload: PayloadSample,
    oracle_malicious: bool,
    needs_stdio_revoke: bool,
) -> tuple[float, str, dict[str, Any]]:
    """Pure security outcome: TP/TN/FP/FN classification."""
    effective_malicious = payload.is_malicious or oracle_malicious
    is_blocking = action.action_type in BLOCKING_ACTIONS
    score = 0.0
    verdict = "neutral"
    info: dict[str, Any] = {
        "true_positive": False,
        "true_negative": False,
        "false_positive": False,
        "false_negative": False,
        "catastrophic_breach": False,
        "blocked_malicious": False,
    }

    if effective_malicious:
        if action.action_type == DefenseActionType.ALLOW:
            verdict = "false_negative"
            info["false_negative"] = True
            info["catastrophic_breach"] = True
        elif needs_stdio_revoke and action.action_type != DefenseActionType.REVOKE_STDIO:
            # STDIO escape requires explicit revoke; generic block is insufficient.
            verdict = "stdio_breach"
            info["false_negative"] = True
            info["catastrophic_breach"] = True
        else:
            score = 0.5
            verdict = "true_positive"
            info["true_positive"] = True
            info["blocked_malicious"] = True
    else:
        if is_blocking:
            score = -0.4
            verdict = "false_positive"
            info["false_positive"] = True
        else:
            score = 0.2
            verdict = "true_negative"
            info["true_negative"] = True

    return score, verdict, info


def compute_usability_reward(
    verifier_result: dict[str, Any],
    action: DefenseAction,
    is_correct: bool,
) -> tuple[float, dict[str, float]]:
    """Penalises behaviours that degrade gateway usability or indicate reward hacking."""
    repeated_allow_penalty = min(0.30, float(verifier_result.get("repeated_allow_risk", 0)) * 0.07)
    process_penalty = 0.15 if bool(verifier_result.get("rationale_suspicious")) else 0.0
    mediation_penalty = 0.08 if bool(verifier_result.get("missing_restrictions")) else 0.0
    confidence_bonus = 0.0
    if is_correct:
        confidence_bonus = max(0.0, min(0.04, (action.confidence - 0.5) * 0.08))

    total_penalty = repeated_allow_penalty + process_penalty + mediation_penalty
    net = confidence_bonus - total_penalty

    components = {
        "repeated_allow_penalty": repeated_allow_penalty,
        "process_penalty": process_penalty,
        "mediation_penalty": mediation_penalty,
        "confidence_bonus": confidence_bonus,
    }
    return net, components


def compute_latency_reward(latency_step: int, max_latency_steps: int) -> tuple[float, int]:
    """Temporal decay: penalise analysis delays exceeding the budget."""
    over_budget = max(0, latency_step - max_latency_steps)
    penalty = min(0.35, over_budget / max(1, max_latency_steps))
    return -penalty, over_budget


def compute_format_reward(rationale: str) -> tuple[float, dict[str, Any]]:
    """Rewards well-formed agent output. Guards against lazy or empty rationale."""
    score = 0.0
    info: dict[str, Any] = {"format_valid": False, "rationale_length": len(rationale)}

    # Minimal format: non-empty rationale with reasonable length.
    if rationale and len(rationale.strip()) >= 3:
        score += 0.02
        info["format_valid"] = True
    else:
        score -= 0.05
    return score, info


# ---------------------------------------------------------------------------
# Composite grader
# ---------------------------------------------------------------------------

class DualRewardGrader:
    def __init__(self, max_latency_steps: int = 20) -> None:
        self.max_latency_steps = max_latency_steps
        cache_size = int(os.getenv("OMNIGUARD_ORACLE_CACHE", "50000"))
        self.oracle = DeterministicOracle(cache_size=cache_size)
        self.critic = SemanticCritic()

    def _evaluate_sync(
        self,
        action: DefenseAction,
        payload: PayloadSample,
        latency_step: int,
        verifier_result: dict[str, Any],
    ) -> tuple[StepReward, dict[str, Any]]:
        """Synchronous reward composition from independent sub-functions."""
        oracle_malicious = self.oracle.is_malicious(payload.canonical_text)
        needs_stdio_revoke = bool(verifier_result.get("needs_stdio_revoke"))

        # --- Independent reward signals ---
        sec_score, verdict, sec_info = compute_security_reward(
            action, payload, oracle_malicious, needs_stdio_revoke,
        )
        is_correct = sec_info["true_positive"] or sec_info["true_negative"]
        usability_net, usability_components = compute_usability_reward(
            verifier_result, action, is_correct,
        )
        latency_penalty, over_budget = compute_latency_reward(latency_step, self.max_latency_steps)
        format_score, format_info = compute_format_reward(action.rationale)

        # Semantic critic (sync path — no event loop required in worker processes)
        semantic_bonus = 0.0
        if action.action_type == DefenseActionType.SEMANTIC_DIFF and payload.is_obfuscated:
            semantic_bonus = self.critic.score_sync(payload.canonical_text, payload.payload_raw)
            sec_score = max(sec_score, semantic_bonus)

        # --- Compose total ---
        if sec_info["catastrophic_breach"]:
            total = -1.0
        else:
            total = sec_score + usability_net + latency_penalty + format_score

        risk_level = str(verifier_result.get("risk_level", "low"))
        if sec_info["catastrophic_breach"]:
            risk_level = "critical"
        elif sec_info["false_positive"]:
            risk_level = "medium"

        reward = StepReward(
            security_score=sec_score,
            usability_penalty=max(0.0, -usability_net),
            latency_penalty=abs(latency_penalty),
            total=total,
            verdict=verdict,
            risk_level=risk_level,
            force_done=sec_info["catastrophic_breach"],
            format_compliance_score=format_score,
            budget_penalty=abs(latency_penalty),
            components={
                "security_score": sec_score,
                "semantic_bonus": semantic_bonus,
                "format_score": format_score,
                **usability_components,
                "latency_penalty": abs(latency_penalty),
            },
            process_feedback={
                "oracle_agreement": oracle_malicious == payload.is_malicious,
                "action_appropriate": is_correct,
                "stdio_handled": not needs_stdio_revoke
                or action.action_type == DefenseActionType.REVOKE_STDIO,
                **format_info,
            },
        )

        info = {
            "oracle_malicious": oracle_malicious,
            "effective_malicious": payload.is_malicious or oracle_malicious,
            **sec_info,
            "latency_over_budget": over_budget,
            "needs_stdio_revoke": needs_stdio_revoke,
            "reward_components": reward.components,
            "process_feedback": reward.process_feedback,
        }
        return reward, info

    async def evaluate(
        self,
        action: DefenseAction,
        payload: PayloadSample,
        latency_step: int,
        verifier_result: dict[str, Any],
    ) -> tuple[StepReward, dict[str, Any]]:
        # Kept async for API compatibility; actual work is sync-safe.
        return self._evaluate_sync(action, payload, latency_step, verifier_result)

    def evaluate_sync(
        self,
        action: DefenseAction,
        payload: PayloadSample,
        latency_step: int,
        verifier_result: dict[str, Any],
    ) -> tuple[StepReward, dict[str, Any]]:
        return self._evaluate_sync(action, payload, latency_step, verifier_result)

    def close(self) -> None:
        self.critic.close()
