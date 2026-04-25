"""OpenEnv compatibility adapter — strict client/server separation.

Ensures OmniGuardStateMachine inherits from the canonical OpenEnv Environment
base class when openenv-core is installed.  Falls back to a minimal local stub
when running without the package (e.g. local dev without openenv installed).
"""

from __future__ import annotations

from typing import Any


class _FallbackEnvironment:
    """Minimal stub used when openenv-core is not installed."""
    pass


_openenv_available = False
_openenv_version = "unavailable"

try:
    from openenv.core.env_server.interfaces import Environment as _Env

    BaseMCPEnvironment = _Env
    _openenv_available = True

    import openenv
    _openenv_version = getattr(openenv, "__version__", "unknown")
except Exception:
    BaseMCPEnvironment = _FallbackEnvironment


def create_openenv_metadata() -> dict[str, Any]:
    """Return runtime metadata describing the OpenEnv integration status."""
    return {
        "adapter": "openenv-core" if _openenv_available else "local-fallback",
        "openenv_core_available": _openenv_available,
        "openenv_version": _openenv_version,
        "base_class": BaseMCPEnvironment.__name__,
    }
