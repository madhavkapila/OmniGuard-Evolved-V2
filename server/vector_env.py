from __future__ import annotations

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from server.env import OmniGuardStateMachine
from server.models import (
    DefenseAction,
    DefenseActionBatch,
    ResetBatchRequest,
    ResetBatchResponse,
    ResetRequestItem,
    StepBatchResponse,
    StepResult,
    ThreatObservation,
)


_WORKER_ENV: OmniGuardStateMachine | None = None


def _worker_init(env_id: int, config: dict[str, Any]) -> None:
    global _WORKER_ENV
    _WORKER_ENV = OmniGuardStateMachine(
        env_id=env_id,
        queue_size=config["queue_size"],
        max_latency_steps=config["max_latency_steps"],
        episode_length=config["episode_length"],
        redis_url=config.get("redis_url"),
        seed=config.get("seed"),
    )


def _worker_ping() -> int:
    if _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")
    return _WORKER_ENV.queue_depth()


def _worker_reset(task_name: str) -> dict[str, Any]:
    if _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")
    obs = _WORKER_ENV.reset(task_name)
    return obs.model_dump(mode="python")


def _worker_step(action_payload: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")
    action = DefenseAction.model_validate(action_payload)
    obs, reward, done, info = _WORKER_ENV.step(action)
    return {
        "env_id": _WORKER_ENV.env_id,
        "observation": obs.model_dump(mode="python"),
        "reward": reward.model_dump(mode="python"),
        "done": done,
        "info": info,
    }


def _worker_metrics() -> dict[str, Any]:
    if _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")
    return _WORKER_ENV.metrics()


def _worker_shutdown() -> bool:
    if _WORKER_ENV is not None:
        _WORKER_ENV.shutdown()
    return True


class AsyncVectorEnvManager:
    def __init__(
        self,
        num_envs: int = 32,
        queue_size: int = 1000,
        max_latency_steps: int = 20,
        episode_length: int = 16,
    ) -> None:
        self.num_envs = num_envs
        self._started = False
        self._worker_timeout_sec = float(os.getenv("OMNIGUARD_WORKER_TIMEOUT_SEC", "120"))
        self._mp_context = multiprocessing.get_context("spawn")
        self._executors: list[ProcessPoolExecutor] = []
        self._config = {
            "queue_size": queue_size,
            "max_latency_steps": max_latency_steps,
            "episode_length": episode_length,
            "redis_url": os.getenv("OMNIGUARD_REDIS_URL"),
            "seed": int(os.getenv("OMNIGUARD_BASE_SEED", "3407")),
        }

    async def startup(self) -> None:
        if self._started:
            return
        loop = asyncio.get_running_loop()
        for env_id in range(self.num_envs):
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=self._mp_context,
                initializer=_worker_init,
                initargs=(env_id, self._config),
            )
            self._executors.append(executor)
        probe_tasks = [loop.run_in_executor(ex, _worker_ping) for ex in self._executors]
        probe_results = await asyncio.gather(*probe_tasks, return_exceptions=True)
        failures = [item for item in probe_results if isinstance(item, Exception)]
        if failures:
            await self.shutdown()
            raise RuntimeError(f"vector env startup failed for {len(failures)} workers")
        self._started = True

    def _validate_env_id(self, env_id: int) -> None:
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} out of range [0, {self.num_envs - 1}]")

    async def reset_batch(self, request: ResetBatchRequest) -> ResetBatchResponse:
        if not self._started:
            raise RuntimeError("vector env not started")

        items = request.items
        if not items:
            items = [ResetRequestItem(env_id=env_id, task_name="default") for env_id in range(self.num_envs)]

        loop = asyncio.get_running_loop()
        pending: list[tuple[int, asyncio.Future]] = []
        for item in items:
            self._validate_env_id(item.env_id)
            future = loop.run_in_executor(
                self._executors[item.env_id],
                _worker_reset,
                item.task_name,
            )
            pending.append((item.env_id, future))

        observations = []
        for _, future in pending:
            payload = await asyncio.wait_for(future, timeout=self._worker_timeout_sec)
            observations.append(ThreatObservation.model_validate(payload))
        return ResetBatchResponse(observations=observations)

    async def step_batch(self, batch: DefenseActionBatch) -> StepBatchResponse:
        if not self._started:
            raise RuntimeError("vector env not started")
        loop = asyncio.get_running_loop()

        pending: list[asyncio.Future] = []
        for action in batch.actions:
            self._validate_env_id(action.env_id)
            action_payload = action.model_dump(mode="python")
            pending.append(
                loop.run_in_executor(
                    self._executors[action.env_id],
                    _worker_step,
                    action_payload,
                )
            )

        raw_results = await asyncio.wait_for(
            asyncio.gather(*pending),
            timeout=self._worker_timeout_sec,
        )
        results = [StepResult.model_validate(item) for item in raw_results]
        return StepBatchResponse(results=results)

    async def health(self) -> dict[int, int]:
        if not self._started:
            return {}
        loop = asyncio.get_running_loop()
        depths = await asyncio.gather(*(loop.run_in_executor(ex, _worker_ping) for ex in self._executors))
        return {env_id: int(depth) for env_id, depth in enumerate(depths)}

    async def metrics(self) -> list[dict[str, Any]]:
        if not self._started:
            return []
        loop = asyncio.get_running_loop()
        data = await asyncio.gather(*(loop.run_in_executor(ex, _worker_metrics) for ex in self._executors))
        return list(data)

    async def shutdown(self) -> None:
        if not self._started:
            for executor in self._executors:
                executor.shutdown(wait=False, cancel_futures=True)
            self._executors.clear()
            return
        loop = asyncio.get_running_loop()
        await asyncio.gather(*(loop.run_in_executor(ex, _worker_shutdown) for ex in self._executors))
        for executor in self._executors:
            executor.shutdown(wait=False, cancel_futures=True)
        self._executors.clear()
        self._started = False
