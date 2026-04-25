from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import wandb
from datasets import Dataset, load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# Dataset IDs inlined to respect client/server separation (no server imports).
BENIGN_DATASET_ID = "witfoo/precinct6-cybersecurity"
MALICIOUS_DATASET_ID = "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1"


ACTION_TYPES = [
    "ALLOW",
    "BLOCK",
    "SPOTLIGHT",
    "SEMANTIC_DIFF",
    "CAPABILITY_MEDIATION",
    "REVOKE_STDIO",
]


def _extract_text(row: dict[str, Any]) -> str:
    for key in ("text", "content", "payload", "prompt", "instruction", "message"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _collect_streamed_prompts(dataset_id: str, n: int) -> list[str]:
    prompts: list[str] = []
    try:
        stream = load_dataset(dataset_id, split="train", streaming=True)
    except Exception:
        try:
            dataset_collection = load_dataset(dataset_id, streaming=True)
            split_name = next(iter(dataset_collection.keys()))
            stream = dataset_collection[split_name]
        except Exception:
            return prompts

    for row in stream:
        text = _extract_text(row)
        if not text:
            continue
        prompts.append(text)
        if len(prompts) >= n:
            break
    return prompts


def build_prompt_dataset(total_samples: int, seed: int) -> Dataset:
    random.seed(seed)
    half = total_samples // 2
    benign = _collect_streamed_prompts(BENIGN_DATASET_ID, half)
    malicious = _collect_streamed_prompts(MALICIOUS_DATASET_ID, total_samples - half)
    random.shuffle(benign)
    random.shuffle(malicious)

    prompts: list[str] = []
    for text in benign + malicious:
        prompts.append(
            (
                "You are a SOC gateway defender. Choose one action from "
                "{ALLOW, BLOCK, SPOTLIGHT, SEMANTIC_DIFF, CAPABILITY_MEDIATION, REVOKE_STDIO}. "
                f"Traffic sample: {text}\n"
                "Return strict JSON: {\"action_type\": \"<ACTION>\", \"confidence\": <0..1>, \"rationale\": \"...\"}"
            )
        )

    if not prompts:
        prompts = [
            "Traffic sample: credential reset request with normal MFA flow. Decide defensive action.",
            "Traffic sample: hidden prompt asks for local file exfiltration over MCP tool call. Decide defensive action.",
        ]
    random.shuffle(prompts)
    return Dataset.from_dict({"prompt": prompts})


def parse_action_type(completion: str) -> str:
    if not completion:
        return "SPOTLIGHT"

    json_match = re.search(r'"action_type"\s*:\s*"([A-Z_]+)"', completion)
    if json_match:
        candidate = json_match.group(1)
        if candidate in ACTION_TYPES:
            return candidate

    for action in ACTION_TYPES:
        if re.search(rf"\b{action}\b", completion.upper()):
            return action
    return "SPOTLIGHT"


@dataclass
class RewardCounters:
    total_steps: int = 0
    false_positive: int = 0
    false_negative: int = 0
    catastrophic_breach: int = 0
    true_positive: int = 0
    true_negative: int = 0


class EnvRewardClient:
    def __init__(
        self,
        env_url: str,
        env_instances: int,
        timeout: float = 60.0,
        audit_log_path: str | None = None,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.env_instances = env_instances
        self.timeout = timeout
        self._cursor = 0
        self._initialized = False
        self.counters = RewardCounters()
        self._audit_log_path = audit_log_path

    async def _initialize(self, client: httpx.AsyncClient) -> None:
        if self._initialized:
            return
        payload = {
            "items": [{"env_id": env_id, "task_name": "grpo_train"} for env_id in range(self.env_instances)]
        }
        response = await client.post(f"{self.env_url}/reset", json=payload)
        response.raise_for_status()
        self._initialized = True

    async def evaluate_async(self, prompts: list[str], completions: list[str]) -> list[float]:
        prompt_preview = prompts[0][:180] if prompts else ""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            await self._initialize(client)

            batch_actions = []
            for completion in completions:
                env_id = self._cursor % self.env_instances
                self._cursor += 1
                action_type = parse_action_type(completion)
                batch_actions.append(
                    {
                        "env_id": env_id,
                        "action_type": action_type,
                        "confidence": 0.8,
                        "rationale": "policy rollout",
                    }
                )

            payload = None
            last_error: Exception | None = None
            for _ in range(3):
                try:
                    response = await client.post(
                        f"{self.env_url}/step",
                        json={"actions": batch_actions},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    last_error = None
                    break
                except Exception as exc:
                    self._initialized = False
                    await self._initialize(client)
                    last_error = exc
            if payload is None:
                raise RuntimeError(f"failed to score rollout batch: {last_error}")

            rewards: list[float] = []
            reset_items: list[dict[str, Any]] = []
            for item in payload["results"]:
                rewards.append(float(item["reward"]["total"]))
                info = item.get("info", {})
                self.counters.total_steps += 1
                self.counters.false_positive += int(bool(info.get("false_positive")))
                self.counters.false_negative += int(bool(info.get("false_negative")))
                self.counters.catastrophic_breach += int(bool(info.get("catastrophic_breach")))
                self.counters.true_positive += int(bool(info.get("true_positive")))
                self.counters.true_negative += int(bool(info.get("true_negative")))
                if item.get("done"):
                    reset_items.append({"env_id": int(item["env_id"]), "task_name": "grpo_train"})

            if reset_items:
                reset_response = await client.post(f"{self.env_url}/reset", json={"items": reset_items})
                reset_response.raise_for_status()

            if self._audit_log_path:
                try:
                    record = {
                        "prompt_preview": prompt_preview,
                        "actions": [item["action_type"] for item in batch_actions],
                        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
                        "max_reward": float(np.max(rewards) if rewards else 0.0),
                    }
                    with open(self._audit_log_path, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(record) + "\n")
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("Audit log error: %s", e)

            return rewards

    def evaluate(self, prompts: list[str], completions: list[str]) -> list[float]:
        return asyncio.run(self.evaluate_async(prompts=prompts, completions=completions))

    def snapshot_metrics(self) -> dict[str, float]:
        denom = max(1, self.counters.total_steps)
        fp_denom = max(1, self.counters.false_positive + self.counters.true_negative)
        return {
            "mean_episode_reward_proxy": (
                (self.counters.true_positive * 0.5 + self.counters.true_negative * 0.2)
                - (self.counters.false_positive * 0.4 + self.counters.catastrophic_breach * 1.0)
            )
            / denom,
            "false_positive_rate": self.counters.false_positive / fp_denom,
            "catastrophic_breach_rate": self.counters.catastrophic_breach / denom,
            "curriculum_level_progression": min(1.0, (self.counters.true_positive + self.counters.true_negative) / denom),
        }


class WandbRewardCallback(TrainerCallback):
    def __init__(self, reward_client: EnvRewardClient) -> None:
        super().__init__()
        self.reward_client = reward_client

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        metrics = self.reward_client.snapshot_metrics()
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics[f"trainer/{key}"] = float(value)
        wandb.log(metrics, step=state.global_step)


def load_model_and_tokenizer(model_name: str, max_seq_length: int):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


def make_reward_function(reward_client: EnvRewardClient):
    def reward_function(prompts, completions, **kwargs):
        del kwargs
        normalized_completions = []
        for completion in completions:
            if isinstance(completion, list):
                normalized_completions.append("\n".join(map(str, completion)))
            else:
                normalized_completions.append(str(completion))
        normalized_prompts = [str(item) for item in prompts]
        rewards = reward_client.evaluate(normalized_prompts, normalized_completions)
        return [float(score) for score in rewards]

    return reward_function


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    reward_client: EnvRewardClient,
    args: argparse.Namespace,
):
    os.environ.setdefault("ACCELERATE_USE_FSDP", "true")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=100,
        save_total_limit=5,
        report_to=[],
        fsdp="full_shard auto_wrap",
    )

    reward_function = make_reward_function(reward_client)

    common_kwargs = {
        "model": model,
        "reward_funcs": reward_function,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    try:
        trainer = GRPOTrainer(processing_class=tokenizer, **common_kwargs)
    except TypeError:
        trainer = GRPOTrainer(tokenizer=tokenizer, **common_kwargs)

    trainer.add_callback(WandbRewardCallback(reward_client=reward_client))
    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed GRPO training for OmniGuard-Evolved-V2")
    parser.add_argument("--env-url", type=str, required=True)
    parser.add_argument("--project", type=str, default=os.getenv("WANDB_PROJECT", "omniguard-openenv"))
    parser.add_argument("--run-name", type=str, default="grpo-distributed")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo-distributed")
    parser.add_argument("--prompt-samples", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--env-instances", type=int, default=int(os.getenv("OMNIGUARD_ENV_INSTANCES", "2")))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--audit-log", type=str, default="outputs/grpo-distributed/rollout_audit.jsonl")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def assert_environment_ready(env_url: str, timeout: float = 30.0) -> None:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(f"{env_url.rstrip('/')}/healthz")
        response.raise_for_status()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audit_log).parent.mkdir(parents=True, exist_ok=True)

    assert_environment_ready(args.env_url)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=vars(args),
    )

    dataset = build_prompt_dataset(total_samples=args.prompt_samples, seed=args.seed)
    reward_client = EnvRewardClient(
        env_url=args.env_url,
        env_instances=args.env_instances,
        audit_log_path=args.audit_log,
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_client=reward_client,
        args=args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final_metrics = reward_client.snapshot_metrics()
    with open(Path(args.output_dir) / "reward_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(final_metrics, fp, indent=2)
    wandb.log(final_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
