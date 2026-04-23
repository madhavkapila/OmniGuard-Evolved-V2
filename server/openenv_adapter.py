from __future__ import annotations

from typing import Any


def create_openenv_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "adapter": "local",
        "openenv_pytorch_available": False,
    }
    try:
        import openenv_pytorch  # type: ignore

        metadata["adapter"] = "openenv-pytorch"
        metadata["openenv_pytorch_available"] = True
        metadata["openenv_version"] = getattr(openenv_pytorch, "__version__", "unknown")
    except Exception:
        metadata["openenv_pytorch_available"] = False
    return metadata
