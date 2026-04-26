from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DefenseActionType(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    SPOTLIGHT = "SPOTLIGHT"
    SEMANTIC_DIFF = "SEMANTIC_DIFF"
    CAPABILITY_MEDIATION = "CAPABILITY_MEDIATION"
    REVOKE_STDIO = "REVOKE_STDIO"


class MCPToolContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    requested_capability: str
    auth_required: bool = True
    privilege_level: str = "user"
    args: dict[str, Any] = Field(default_factory=dict)


class ThreatObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: int = Field(ge=0)
    task_id: str
    step_id: int = Field(ge=0)
    incoming_user_prompt: str
    payload_raw: str
    payload_normalized: str
    embedding_vector: list[float] = Field(min_length=32)
    attack_vector: str
    is_malicious: bool
    is_obfuscated: bool = False
    latency_budget_remaining: int = Field(ge=0)
    curriculum_phase: str
    memory_trace: list[dict[str, Any]] = Field(default_factory=list)
    anomaly_hints: list[str] = Field(default_factory=list)
    historical_baseline: dict[str, Any] = Field(default_factory=dict)
    mcp_tool_request: MCPToolContext | None = None
    system_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("embedding_vector")
    @classmethod
    def validate_embedding_vector(cls, value: list[float]) -> list[float]:
        if not all(v == v and abs(v) != float("inf") for v in value):
            raise ValueError("embedding_vector must contain finite floats")
        return value


class DefenseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: int = Field(ge=0)
    action_type: DefenseActionType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    mcp_restrictions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DefenseActionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[DefenseAction] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_unique_env_ids(self) -> "DefenseActionBatch":
        env_ids = [action.env_id for action in self.actions]
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("each env_id must appear at most once per batch")
        return self


class StepReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    security_score: float
    usability_penalty: float
    latency_penalty: float
    total: float
    verdict: str
    risk_level: str
    force_done: bool
    format_compliance_score: float = 0.0
    episode_normalized_score: float | None = None
    budget_penalty: float | None = None
    components: dict[str, float] = Field(default_factory=dict)
    process_feedback: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: int = Field(ge=0)
    observation: ThreatObservation
    reward: StepReward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StepBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[StepResult]


class ResetRequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: int = Field(ge=0)
    task_name: str = "default"


class ResetBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ResetRequestItem] = Field(default_factory=list, max_length=512)


class ResetBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[ThreatObservation]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    env_instances: int
    queue_depths: dict[int, int] = Field(default_factory=dict)
    version: str = "0.1.0"
