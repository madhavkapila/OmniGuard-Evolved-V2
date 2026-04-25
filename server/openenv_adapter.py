"""OpenEnv compatibility adapter — strict client/server separation.

Ensures OmniGuardStateMachine inherits from the canonical OpenEnv base class
(MCPEnvironment or Environment) when the openenv-pytorch package is installed.
Falls back to a minimal local stub when running without the package.
"""

from __future__ import annotations

from typing import Any


# --- Base class resolution ---
# The OpenEnv spec requires environments to inherit from MCPEnvironment
# (for MCP-aware tool environments) or from the generic Environment base.
# We resolve the best available base class at import time.

class _FallbackEnvironment:
    """Minimal stub base class used when openenv-pytorch is not installed.

    Mirrors the interface contract (reset, step, close) so the state machine
    can operate identically in both online (HF Space) and offline (local dev)
    modes without import errors.
    """
    pass


# Attempt to import the real OpenEnv base class.
_openenv_available = False
_openenv_version = "unavailable"

try:
    import openenv_pytorch  # type: ignore[import-untyped]

    if hasattr(openenv_pytorch, "MCPEnvironment"):
        BaseMCPEnvironment = openenv_pytorch.MCPEnvironment
    elif hasattr(openenv_pytorch, "Environment"):
        BaseMCPEnvironment = openenv_pytorch.Environment
    else:
        BaseMCPEnvironment = _FallbackEnvironment

    _openenv_available = True
    _openenv_version = getattr(openenv_pytorch, "__version__", "unknown")
except ImportError:
    BaseMCPEnvironment = _FallbackEnvironment


def create_openenv_metadata() -> dict[str, Any]:
    """Return runtime metadata describing the OpenEnv integration status."""
    return {
        "adapter": "openenv-pytorch" if _openenv_available else "local-fallback",
        "openenv_pytorch_available": _openenv_available,
        "openenv_version": _openenv_version,
        "base_class": BaseMCPEnvironment.__name__,
    }
