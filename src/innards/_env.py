"""Versions and host facts recorded with every run, and the device helpers ``observe`` uses."""

from __future__ import annotations

import os
import platform
import socket
from typing import Any

__all__ = ["DEFAULT_BUDGET_GB", "budget_bytes", "device_name", "environment", "host_kind"]

DEFAULT_BUDGET_GB = 16.0
"""The memory budget Innards checks against (GPU memory, or unified memory on Apple silicon), unless
``budget_gb`` or ``$INNARDS_GPU_BUDGET_GB`` says otherwise. 16 GB fits a local machine running models at 16 bit."""


def budget_bytes(budget_gb: float | None = None) -> int:
    value = budget_gb if budget_gb is not None else float(os.environ.get("INNARDS_GPU_BUDGET_GB") or DEFAULT_BUDGET_GB)
    if value <= 0:
        raise ValueError("The memory budget must be positive")
    return int(value * 1e9)


def host_kind() -> str:
    if os.environ.get("INNARDS_HOST"):
        return os.environ["INNARDS_HOST"]
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("SM_CURRENT_HOST") or os.environ.get("SAGEMAKER_INTERNAL_IMAGE_URI"):
        return "sagemaker"
    if os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("ECS_CONTAINER_METADATA_URI"):
        return "aws"
    return "local"


def device_name(device: Any) -> str:
    """``NVIDIA H200``, ``mps (arm64)`` or ``cpu (x86_64)``."""
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        try:
            import torch

            index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
            return str(torch.cuda.get_device_name(index))
        except Exception:
            return "cuda"
    return f"{kind} ({platform.machine()})"


def environment() -> dict[str, Any]:
    """Versions and host facts, recorded once per run and per conversation."""
    from innards import __version__

    env: dict[str, Any] = {
        "innards": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": socket.gethostname(),
        "host_kind": host_kind(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = bool(torch.cuda.is_available())
        env["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env
