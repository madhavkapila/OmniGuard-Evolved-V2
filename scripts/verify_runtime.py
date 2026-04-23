from __future__ import annotations

import argparse
import compileall
import importlib
import sys
from pathlib import Path

import httpx


REQUIRED_IMPORTS = [
    "fastapi",
    "pydantic",
    "datasets",
    "numpy",
    "httpx",
    "torch",
    "transformers",
    "trl",
    "accelerate",
    "wandb",
]

OPTIONAL_IMPORTS = [
    "unsloth",
    "openenv_pytorch",
    "peft",
    "redis",
]


def check_python_version() -> tuple[bool, str]:
    ok = sys.version_info >= (3, 12)
    msg = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return ok, msg


def check_imports(module_names: list[str]) -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            passed.append(module_name)
        except Exception:
            failed.append(module_name)
    return passed, failed


def check_expected_files(root: Path) -> list[str]:
    required_paths = [
        root / "server" / "app.py",
        root / "server" / "vector_env.py",
        root / "server" / "generator.py",
        root / "training" / "grpo_distributed.py",
        root / "eval" / "benchmark.py",
        root / "docker-compose.yml",
        root / "pyproject.toml",
        root / ".env.example",
    ]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    return missing


def check_compile(root: Path) -> bool:
    return compileall.compile_dir(
        str(root / "server"),
        quiet=1,
    ) and compileall.compile_dir(
        str(root / "training"),
        quiet=1,
    ) and compileall.compile_dir(
        str(root / "eval"),
        quiet=1,
    ) and compileall.compile_dir(
        str(root / "worker"),
        quiet=1,
    )


def check_live_endpoints(env_url: str) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=10.0) as client:
            health = client.get(f"{env_url.rstrip('/')}/healthz")
            ready = client.get(f"{env_url.rstrip('/')}/readyz")
            if health.status_code == 200 and ready.status_code == 200:
                return True, "healthz=200 readyz=200"
            return False, f"healthz={health.status_code} readyz={ready.status_code}"
    except Exception as exc:
        return False, f"endpoint check failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify OmniGuard runtime readiness")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--env-url", type=str, default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    failures: list[str] = []

    py_ok, py_msg = check_python_version()
    print(f"[python] {py_msg}")
    if not py_ok:
        failures.append("python version must be >= 3.12")

    passed_required, failed_required = check_imports(REQUIRED_IMPORTS)
    print(f"[imports-required] ok={passed_required}")
    if failed_required:
        print(f"[imports-required] missing={failed_required}")
        failures.append("missing required imports")

    passed_optional, failed_optional = check_imports(OPTIONAL_IMPORTS)
    print(f"[imports-optional] ok={passed_optional}")
    if failed_optional:
        print(f"[imports-optional] missing={failed_optional}")

    missing_files = check_expected_files(root)
    if missing_files:
        print(f"[files] missing={missing_files}")
        failures.append("missing expected files")
    else:
        print("[files] all expected files present")

    compile_ok = check_compile(root)
    print(f"[compile] ok={compile_ok}")
    if not compile_ok:
        failures.append("compile check failed")

    if args.env_url:
        endpoint_ok, endpoint_msg = check_live_endpoints(args.env_url)
        print(f"[endpoints] {endpoint_msg}")
        if not endpoint_ok:
            failures.append("live endpoint check failed")

    if failures:
        print(f"[summary] failures={failures}")
        if args.strict:
            raise SystemExit(1)
        raise SystemExit(0)

    print("[summary] runtime verification passed")


if __name__ == "__main__":
    main()
