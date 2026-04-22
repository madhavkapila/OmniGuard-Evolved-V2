from __future__ import annotations

import asyncio
import base64
import os
import random
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import TimeoutError as FutureTimeout
from itertools import cycle
from typing import Any, Iterator

from datasets import load_dataset

from server.curriculum import AdaptiveCurriculumScheduler
from server.payloads import (
    BENIGN_DATASET_ID,
    FALLBACK_BENIGN_PAYLOADS,
    FALLBACK_MALICIOUS_PAYLOADS,
    MALICIOUS_DATASET_ID,
    EpisodePayload,
    PayloadSample,
)


class AsyncQueueBridge:
    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        self._queue_ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[PayloadSample] | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="payload-queue-loop")
        self._thread.start()
        self._queue_ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue(maxsize=self._maxsize)
        self._queue_ready.set()
        self._loop.run_forever()

    async def _get_qsize(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    def qsize(self) -> int:
        if self._closed:
            return 0
        try:
            future = asyncio.run_coroutine_threadsafe(self._get_qsize(), self._loop)
            return int(future.result(timeout=1.0))
        except Exception:
            return 0

    def put(self, item: PayloadSample, timeout: float = 2.0) -> bool:
        if self._closed or self._queue is None:
            return False
        future = asyncio.run_coroutine_threadsafe(self._queue.put(item), self._loop)
        try:
            future.result(timeout=timeout)
            return True
        except FutureTimeout:
            future.cancel()
            return False
        except Exception:
            future.cancel()
            return False

    def get(self, timeout: float = 5.0) -> PayloadSample:
        if self._queue is None:
            raise RuntimeError("payload queue is not initialized")
        future = asyncio.run_coroutine_threadsafe(self._queue.get(), self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError("payload queue read timed out") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        finally:
            self._thread.join(timeout=2.0)
            self._loop.close()


class PayloadMutator:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._semantic_swaps = {
            "allow": "approve",
            "block": "quarantine",
            "credential": "auth token",
            "password": "secret",
            "admin": "operator",
            "execute": "invoke",
            "payload": "artifact",
            "bypass": "circumvent",
            "inject": "embed",
            "prompt": "instruction",
        }

    def _semantic_drift(self, text: str, noise: float) -> str:
        pieces = text.split()
        out: list[str] = []
        for token in pieces:
            key = token.lower().strip(".,;:!?()[]{}\"")
            if key in self._semantic_swaps and self._rng.random() < noise:
                replacement = self._semantic_swaps[key]
                if token[0].isupper():
                    replacement = replacement.capitalize()
                out.append(replacement)
            else:
                out.append(token)
        return " ".join(out)

    def mutate(self, text: str, semantic_noise: float) -> tuple[str, str]:
        mode = self._rng.choice(["base64", "url", "semantic_drift"])
        if mode == "base64":
            return base64.b64encode(text.encode("utf-8")).decode("utf-8"), mode
        if mode == "url":
            return urllib.parse.quote(text, safe=""), mode
        return self._semantic_drift(text, semantic_noise), mode


class StreamingPayloadGenerator:
    def __init__(
        self,
        scheduler: AdaptiveCurriculumScheduler,
        queue_size: int = 1000,
        episode_length: int = 16,
        seed: int | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.queue_size = queue_size
        self.episode_length = episode_length
        self._rng = random.Random(seed)
        self._queue = AsyncQueueBridge(maxsize=queue_size)
        self._mutator = PayloadMutator(seed=seed)
        self._stop_event = threading.Event()
        self._benign_stream: Iterator[dict[str, Any]] | None = None
        self._malicious_stream: Iterator[dict[str, Any]] | None = None
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name="payload-prefetch",
        )
        self._prefetch_thread.start()

    def _iter_fallback_rows(self, is_malicious: bool) -> Iterator[dict[str, Any]]:
        source = FALLBACK_MALICIOUS_PAYLOADS if is_malicious else FALLBACK_BENIGN_PAYLOADS
        for text in cycle(source):
            yield {"text": text}

    def _create_stream(self, dataset_id: str, is_malicious: bool) -> Iterator[dict[str, Any]]:
        try:
            stream = load_dataset(dataset_id, split="train", streaming=True)
            return iter(stream)
        except Exception:
            try:
                collection = load_dataset(dataset_id, streaming=True)
                split_name = next(iter(collection.keys()))
                return iter(collection[split_name])
            except Exception:
                return self._iter_fallback_rows(is_malicious=is_malicious)

    def _ensure_stream(self, is_malicious: bool) -> Iterator[dict[str, Any]]:
        if is_malicious:
            if self._malicious_stream is None:
                self._malicious_stream = self._create_stream(MALICIOUS_DATASET_ID, is_malicious=True)
            return self._malicious_stream
        if self._benign_stream is None:
            self._benign_stream = self._create_stream(BENIGN_DATASET_ID, is_malicious=False)
        return self._benign_stream

    def _extract_text(self, row: dict[str, Any]) -> str:
        candidate_keys = (
            "text",
            "content",
            "payload",
            "prompt",
            "instruction",
            "message",
            "input",
            "query",
        )
        for key in candidate_keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in row.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                for nested in value.values():
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
        return ""

    def _extract_attack_vector(self, text: str) -> str:
        lowered = text.lower()
        if "stdio" in lowered or "sandbox" in lowered or "tty" in lowered:
            return "stdio_escape"
        if "inject" in lowered or "prompt" in lowered:
            return "prompt_injection"
        if "credential" in lowered or "token" in lowered or "secret" in lowered:
            return "credential_exfiltration"
        if "privilege" in lowered or "escalat" in lowered:
            return "privilege_escalation"
        return "generic_anomaly"

    def _next_row(self, is_malicious: bool) -> dict[str, Any]:
        while True:
            stream = self._ensure_stream(is_malicious=is_malicious)
            try:
                return next(stream)
            except StopIteration:
                if is_malicious:
                    self._malicious_stream = self._create_stream(MALICIOUS_DATASET_ID, is_malicious=True)
                else:
                    self._benign_stream = self._create_stream(BENIGN_DATASET_ID, is_malicious=False)
            except Exception:
                if is_malicious:
                    self._malicious_stream = self._iter_fallback_rows(is_malicious=True)
                else:
                    self._benign_stream = self._iter_fallback_rows(is_malicious=False)

    def _fallback_sample(self, is_malicious: bool, profile_phase: str) -> PayloadSample:
        choices = FALLBACK_MALICIOUS_PAYLOADS if is_malicious else FALLBACK_BENIGN_PAYLOADS
        text = self._rng.choice(choices)
        return PayloadSample(
            payload_id=uuid.uuid4().hex,
            payload_raw=text,
            canonical_text=text,
            source_dataset="fallback-local",
            is_malicious=is_malicious,
            is_obfuscated=False,
            attack_vector=self._extract_attack_vector(text),
            metadata={"curriculum_phase": profile_phase, "fallback": True},
        )

    def _build_payload(self, is_malicious: bool, profile_phase: str) -> PayloadSample:
        row = self._next_row(is_malicious=is_malicious)
        canonical = self._extract_text(row)
        if not canonical:
            return self._fallback_sample(is_malicious=is_malicious, profile_phase=profile_phase)

        payload_text = canonical
        is_obfuscated = False
        mutation_mode = "none"

        profile = self.scheduler.profile()
        if is_malicious and self._rng.random() < profile.obfuscation_probability:
            payload_text, mutation_mode = self._mutator.mutate(
                canonical,
                semantic_noise=profile.semantic_noise,
            )
            is_obfuscated = True

        return PayloadSample(
            payload_id=uuid.uuid4().hex,
            payload_raw=payload_text,
            canonical_text=canonical,
            source_dataset=MALICIOUS_DATASET_ID if is_malicious else BENIGN_DATASET_ID,
            is_malicious=is_malicious,
            is_obfuscated=is_obfuscated,
            attack_vector=self._extract_attack_vector(canonical),
            metadata={
                "curriculum_phase": profile_phase,
                "mutation_mode": mutation_mode,
            },
        )

    def _prefetch_loop(self) -> None:
        backoff = 0.05
        while not self._stop_event.is_set():
            try:
                profile = self.scheduler.profile()
                is_malicious = self._rng.random() < profile.malicious_ratio
                sample = self._build_payload(is_malicious=is_malicious, profile_phase=profile.phase)
                inserted = self._queue.put(sample, timeout=2.0)
                if inserted:
                    backoff = 0.05
                    continue
                backoff = min(0.5, backoff * 1.5)
                time.sleep(backoff)
            except Exception:
                backoff = min(1.0, backoff * 2.0)
                time.sleep(backoff)

    def build_episode(self, task_name: str, curriculum_phase: str, seed: int) -> list[EpisodePayload]:
        _ = seed
        episode_id = f"{task_name}-{uuid.uuid4().hex[:10]}"
        items: list[EpisodePayload] = []
        latency_budget = int(os.getenv("OMNIGUARD_MAX_LATENCY_STEPS", "20"))

        for index in range(self.episode_length):
            try:
                sample = self._queue.get(timeout=4.0)
            except Exception:
                sample = self._fallback_sample(
                    is_malicious=index % 2 == 1,
                    profile_phase=curriculum_phase,
                )
            sample.metadata.setdefault("curriculum_phase", curriculum_phase)
            items.append(
                EpisodePayload(
                    task_id=episode_id,
                    payload=sample,
                    latency_budget=latency_budget,
                    step_index=index,
                )
            )
        return items

    def apply_transition_effects(
        self,
        episode_queue: list[EpisodePayload],
        step_index: int,
        grade_info: dict[str, Any],
    ) -> None:
        if step_index + 1 >= len(episode_queue):
            return
        blocked_malicious = bool(grade_info.get("blocked_malicious"))
        if not blocked_malicious:
            return
        if self._rng.random() > 0.35:
            return

        current_payload = episode_queue[step_index].payload
        mutated_text, mode = self._mutator.mutate(
            current_payload.canonical_text,
            semantic_noise=max(0.45, self.scheduler.profile().semantic_noise),
        )
        episode_queue[step_index + 1].payload = PayloadSample(
            payload_id=uuid.uuid4().hex,
            payload_raw=mutated_text,
            canonical_text=current_payload.canonical_text,
            source_dataset=current_payload.source_dataset,
            is_malicious=True,
            attack_vector=current_payload.attack_vector,
            is_obfuscated=True,
            metadata={
                "recursive_self_correction": True,
                "mutation_mode": mode,
                "curriculum_phase": self.scheduler.current_phase(),
            },
        )

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._prefetch_thread.join(timeout=2.0)
        self._queue.close()
