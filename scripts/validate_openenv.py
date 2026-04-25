#!/usr/bin/env python3
"""validate_openenv.py — Pre-flight compliance checker for OpenEnv.

Validates that the OmniGuard-Evolved-V2 codebase does NOT use reserved tool names
(reset, step, state, close) as MCP tool identifiers, and verifies the openenv.yaml
manifest is well-formed.

Usage:
    python scripts/validate_openenv.py
"""

from __future__ import annotations

import ast
import pathlib
import sys
import yaml

# OpenEnv reserves these names for the Gym-style API surface.
# MCP tools MUST NOT shadow these.
RESERVED_TOOL_NAMES = frozenset({"reset", "step", "state", "close"})

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_ROOT / "server"
MANIFEST_PATH = PROJECT_ROOT / "openenv.yaml"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def check_reserved_tool_names() -> list[str]:
    """Scan all Python files in server/ for string literals matching reserved names
    used as MCP tool identifiers (e.g., tool_name='step')."""
    violations: list[str] = []

    for py_file in SERVER_DIR.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            # Check keyword arguments like tool_name="step"
            if isinstance(node, ast.keyword):
                if node.arg and "tool" in node.arg.lower():
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        if node.value.value.lower() in RESERVED_TOOL_NAMES:
                            violations.append(
                                f"  {py_file.relative_to(PROJECT_ROOT)}:{node.lineno} "
                                f"— reserved tool name '{node.value.value}' used in kwarg '{node.arg}'"
                            )
            # Check dict literals with tool_name keys
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and "tool" in key.value.lower()
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and value.value.lower() in RESERVED_TOOL_NAMES
                    ):
                        violations.append(
                            f"  {py_file.relative_to(PROJECT_ROOT)}:{node.lineno} "
                            f"— reserved tool name '{value.value}' in dict key '{key.value}'"
                        )
    return violations


def check_mcp_tool_definitions() -> list[str]:
    """Check MCPToolContext usages don't clash with reserved names."""
    violations: list[str] = []

    for py_file in SERVER_DIR.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_no, line in enumerate(source.splitlines(), start=1):
            # Quick heuristic: look for tool_name= assignments with reserved strings
            if "tool_name" in line:
                for reserved in RESERVED_TOOL_NAMES:
                    if f'"{reserved}"' in line or f"'{reserved}'" in line:
                        violations.append(
                            f"  {py_file.relative_to(PROJECT_ROOT)}:{line_no} "
                            f"— tool_name set to reserved '{reserved}'"
                        )
    return violations


def check_manifest() -> list[str]:
    """Validate openenv.yaml exists and has required top-level keys."""
    issues: list[str] = []
    if not MANIFEST_PATH.exists():
        issues.append("  openenv.yaml not found at project root")
        return issues

    try:
        with open(MANIFEST_PATH) as f:
            manifest = yaml.safe_load(f)
    except Exception as e:
        issues.append(f"  openenv.yaml parse error: {e}")
        return issues

    required_keys = {"name", "description", "version", "entrypoint", "tasks"}
    missing = required_keys - set(manifest.keys())
    if missing:
        issues.append(f"  openenv.yaml missing required keys: {missing}")

    # Validate tasks have names
    tasks = manifest.get("tasks", [])
    if not tasks:
        issues.append("  openenv.yaml: no tasks defined")
    else:
        for i, task in enumerate(tasks):
            if "name" not in task:
                issues.append(f"  openenv.yaml: task[{i}] missing 'name'")

    return issues


def check_base_class_inheritance() -> list[str]:
    """Verify OmniGuardStateMachine inherits from BaseMCPEnvironment."""
    issues: list[str] = []
    env_path = SERVER_DIR / "env.py"

    if not env_path.exists():
        issues.append("  server/env.py not found")
        return issues

    source = env_path.read_text(encoding="utf-8")
    if "BaseMCPEnvironment" not in source:
        issues.append("  server/env.py: OmniGuardStateMachine does not reference BaseMCPEnvironment")
    if "class OmniGuardStateMachine" not in source:
        issues.append("  server/env.py: OmniGuardStateMachine class not found")

    # Verify import of openenv_adapter
    if "from server.openenv_adapter import" not in source:
        issues.append("  server/env.py: missing import from server.openenv_adapter")

    return issues


def main() -> int:
    print("=" * 60)
    print("  OmniGuard-Evolved-V2 — OpenEnv Compliance Validator")
    print("=" * 60)
    print()

    exit_code = 0

    # 1. Reserved tool names (AST scan)
    print("1. Checking reserved tool names (reset, step, state, close)...")
    violations = check_reserved_tool_names()
    violations += check_mcp_tool_definitions()
    if violations:
        print(f"   {FAIL} Found {len(violations)} violation(s):")
        for v in violations:
            print(f"      {v}")
        exit_code = 1
    else:
        print(f"   {PASS} No reserved tool name collisions found.")

    print()

    # 2. Manifest validation
    print("2. Validating openenv.yaml manifest...")
    manifest_issues = check_manifest()
    if manifest_issues:
        print(f"   {FAIL} Found {len(manifest_issues)} issue(s):")
        for issue in manifest_issues:
            print(f"      {issue}")
        exit_code = 1
    else:
        print(f"   {PASS} openenv.yaml is valid and complete.")

    print()

    # 3. Base class inheritance
    print("3. Checking OpenEnv base class inheritance...")
    inheritance_issues = check_base_class_inheritance()
    if inheritance_issues:
        print(f"   {FAIL} Found {len(inheritance_issues)} issue(s):")
        for issue in inheritance_issues:
            print(f"      {issue}")
        exit_code = 1
    else:
        print(f"   {PASS} OmniGuardStateMachine correctly inherits BaseMCPEnvironment.")

    print()

    # 4. Client/server separation
    print("4. Checking client/server separation...")
    client_violations: list[str] = []
    training_dir = PROJECT_ROOT / "training"
    eval_dir = PROJECT_ROOT / "eval"
    for scan_dir in [training_dir, eval_dir]:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            # Clients must NOT import server internals (except models for type hints)
            bad_imports = [
                "from server.env import",
                "from server.graders import",
                "from server.generator import",
                "from server.verifier import",
                "from server.vector_env import",
            ]
            for bad in bad_imports:
                if bad in source:
                    client_violations.append(
                        f"  {py_file.relative_to(PROJECT_ROOT)} imports server internals: {bad}"
                    )

    if client_violations:
        print(f"   {WARN} Found {len(client_violations)} potential violation(s):")
        for v in client_violations:
            print(f"      {v}")
    else:
        print(f"   {PASS} Client code respects server boundary.")

    print()
    print("=" * 60)
    if exit_code == 0:
        print(f"  {PASS} ALL CHECKS PASSED — Ready for OpenEnv submission.")
    else:
        print(f"  {FAIL} COMPLIANCE ISSUES FOUND — Fix before submission.")
    print("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
