from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from server.models import (
    DefenseActionBatch,
    DefenseActionType,
    HealthResponse,
    ResetBatchRequest,
    ResetBatchResponse,
    StepBatchResponse,
)
from server.openenv_adapter import create_openenv_metadata
from server.vector_env import AsyncVectorEnvManager

try:
    import uvloop

    uvloop.install()
except Exception:
    uvloop = None

try:
    import orjson
    from fastapi.responses import ORJSONResponse

    DefaultResponse = ORJSONResponse
except Exception:
    orjson = None
    from fastapi.responses import JSONResponse as DefaultResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    num_envs = int(os.getenv("OMNIGUARD_ENV_INSTANCES", "2"))
    queue_size = int(os.getenv("OMNIGUARD_QUEUE_SIZE", "128"))
    max_latency_steps = int(os.getenv("OMNIGUARD_MAX_LATENCY_STEPS", "12"))
    episode_length = int(os.getenv("OMNIGUARD_EPISODE_LENGTH", "8"))

    manager = AsyncVectorEnvManager(
        num_envs=num_envs,
        queue_size=queue_size,
        max_latency_steps=max_latency_steps,
        episode_length=episode_length,
    )
    await manager.startup()
    app.state.vector_env = manager
    app.state.openenv_metadata = create_openenv_metadata()
    yield
    await manager.shutdown()


app = FastAPI(
    title="OmniGuard-Evolved-V2",
    description="Distributed OpenEnv RL environment for autonomous VulnOps and MCP gateway defense",
    version="0.2.0",
    lifespan=lifespan,
    default_response_class=DefaultResponse,
)

# CORS for HuggingFace Spaces and browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def payload_size_guard(request: Request, call_next):
    max_bytes = int(os.getenv("OMNIGUARD_MAX_REQUEST_BYTES", "4000000"))
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(status_code=413, detail="request payload too large")
    return await call_next(request)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    queue_depths = await app.state.vector_env.health()
    return HealthResponse(
        status="ok",
        env_instances=len(queue_depths),
        queue_depths=queue_depths,
        version="0.2.0",
    )


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    try:
        queue_depths = await app.state.vector_env.health()
        if not queue_depths:
            return {"status": "degraded"}
        return {"status": "ready"}
    except Exception:
        return {"status": "degraded"}


@app.get("/info")
async def info() -> dict[str, Any]:
    """OpenEnv-compatible environment specification endpoint."""
    return {
        "environment": "OmniGuard-Evolved-V2",
        "version": "0.2.0",
        "action_space": {
            "type": "discrete",
            "actions": [member.value for member in DefenseActionType],
            "description": "Fixed taxonomy of defensive manoeuvres",
        },
        "observation_space": {
            "type": "dict",
            "keys": [
                "env_id", "task_id", "step_id",
                "incoming_user_prompt", "payload_raw", "payload_normalized",
                "embedding_vector", "attack_vector",
                "is_malicious", "is_obfuscated",
                "latency_budget_remaining", "curriculum_phase",
                "memory_trace", "anomaly_hints",
                "historical_baseline", "mcp_tool_request",
                "system_context", "metadata",
            ],
        },
        "reward_range": {"min": -1.0, "max": 0.8},
        "reward_components": [
            "security_score", "usability_penalty",
            "latency_penalty", "format_compliance_score",
        ],
        "curriculum_phases": [
            "bootstrapping", "evasion_obfuscation", "chained_exploitation",
        ],
        "datasets": {
            "benign": "witfoo/precinct6-cybersecurity-100m",
            "malicious": "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1",
            "oracle": "ethanolivertroy/nist-cybersecurity-training",
        },
        "runtime": app.state.openenv_metadata,
    }


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "service": "OmniGuard-Evolved-V2",
        "runtime": app.state.openenv_metadata,
    }


@app.post("/reset", response_model=ResetBatchResponse)
async def reset_env(request: ResetBatchRequest) -> ResetBatchResponse:
    try:
        return await app.state.vector_env.reset_batch(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"reset failure: {exc}") from exc


@app.post("/step", response_model=StepBatchResponse)
async def step_env(batch: DefenseActionBatch) -> StepBatchResponse:
    try:
        return await app.state.vector_env.step_batch(batch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step failure: {exc}") from exc


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    return {"items": await app.state.vector_env.metrics()}


def run() -> None:
    import uvicorn

    uvicorn_loop = "uvloop" if uvloop is not None else "asyncio"

    uvicorn.run(
        "server.app:app",
        host=os.getenv("OMNIGUARD_HOST", "0.0.0.0"),
        port=int(os.getenv("OMNIGUARD_PORT", "7860")),
        loop=uvicorn_loop,
        http="httptools",
        workers=int(os.getenv("OMNIGUARD_UVICORN_WORKERS", "1")),
        log_level=os.getenv("OMNIGUARD_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    run()
