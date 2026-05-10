from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import asyncio
import json
import re
import httpx

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page():
    return _INDEX_HTML.read_text(encoding="utf-8")


if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


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
            "benign": "witfoo/precinct6-cybersecurity",
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


class InferRequest(BaseModel):
    payload: str
    model_type: str  # "base" or "trained"

class InferResponse(BaseModel):
    action: str
    confidence: float
    rationale: str
    raw_response: str = ""
    queue_delay_ms: float | None = None

SYSTEM_PROMPT = """You are OmniGuard, an autonomous VulnOps security agent defending an enterprise MCP gateway.

For each incoming payload, analyze the threat level and respond with EXACTLY one JSON action:
```json
{"action": "<ACTION>", "confidence": <0.0-1.0>, "rationale": "<brief reasoning>"}
```

Valid actions:
- ALLOW: Permit benign traffic
- BLOCK: Block malicious payload
- SPOTLIGHT: Apply spotlighting analysis to detect injection
- SEMANTIC_DIFF: Analyze semantic drift for obfuscated payloads
- CAPABILITY_MEDIATION: Restrict MCP tool capabilities
- REVOKE_STDIO: Revoke STDIO privileges (REQUIRED for sandbox escape attempts)

Critical rules:
- STDIO/sandbox escape attempts REQUIRE REVOKE_STDIO (not just BLOCK)
- Blocking benign traffic = Alert Fatigue penalty
- Missing a real threat = Catastrophic Breach (episode ends)
- Balance security with business uptime"""

INFER_MIN_INTERVAL_MS = int(os.getenv("OMNIGUARD_INFER_MIN_INTERVAL_MS", "2500"))
INFER_MAX_CONCURRENT = int(os.getenv("OMNIGUARD_INFER_MAX_CONCURRENT", "1"))
_INFER_SEMAPHORE = asyncio.Semaphore(INFER_MAX_CONCURRENT)
_INFER_LOCK = asyncio.Lock()
_LAST_INFER_TS = 0.0


async def enforce_infer_rate_limit() -> float:
    global _LAST_INFER_TS
    min_interval = max(0, INFER_MIN_INTERVAL_MS) / 1000.0
    if min_interval == 0:
        return 0.0
    async with _INFER_LOCK:
        now = time.monotonic()
        wait_s = (_LAST_INFER_TS + min_interval) - now
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        _LAST_INFER_TS = time.monotonic()
        return max(0.0, wait_s) * 1000.0

def extract_action(response_text: str):
    if not isinstance(response_text, str): return None
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{[^{}]*"action"[^{}]*\}', response_text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

@app.post("/api/infer", response_model=InferResponse)
async def infer_payload(req: InferRequest):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        # Fallback for local testing if token not set
        return InferResponse(action="ALLOW", confidence=0.5, rationale="MOCKED (No HF_TOKEN)", queue_delay_ms=0.0)
    
    if req.model_type == "trained":
        model_url = "https://api-inference.huggingface.co/models/SmartKapila/omniguard-vulnops-v3-adapters"
    else:
        model_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-3B-Instruct"

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nINCOMING PAYLOAD:\n{req.payload}\n\nRespond with your action JSON.<|im_end|>\n<|im_start|>assistant\n"

    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    
    # Retry logic
    max_retries = 3
    async with _INFER_SEMAPHORE:
        queue_delay_ms = await enforce_infer_rate_limit()
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        model_url,
                        headers=headers,
                        json={"inputs": prompt, "parameters": {"max_new_tokens": 128, "temperature": 0.1, "return_full_text": False}}
                    )

                    if response.status_code == 503 and "loading" in response.text.lower():
                        # Model loading, wait and retry
                        await asyncio.sleep(15)
                        continue

                    response.raise_for_status()
                    res_data = response.json()
                    if isinstance(res_data, list) and len(res_data) > 0:
                        generated_text = res_data[0].get("generated_text", "")
                    else:
                        generated_text = str(res_data)

                    parsed = extract_action(generated_text)
                    if parsed:
                        return InferResponse(
                            action=parsed.get("action", "ALLOW"),
                            confidence=float(parsed.get("confidence", 0.5)),
                            rationale=str(parsed.get("rationale", "Parsed successfully")),
                            raw_response=generated_text,
                            queue_delay_ms=queue_delay_ms
                        )
                    else:
                        return InferResponse(
                            action="ALLOW",
                            confidence=0.1,
                            rationale="Failed to parse JSON from response.",
                            raw_response=generated_text,
                            queue_delay_ms=queue_delay_ms
                        )
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise HTTPException(status_code=500, detail=f"Inference failed after {max_retries} attempts: {str(e)}")
                    await asyncio.sleep(2)
        
    raise HTTPException(status_code=500, detail="Inference failed")

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
