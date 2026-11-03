"""Checks before a model is downloaded, and again after it has loaded.

Every failure says what is wrong, gives the numbers and says how to fix it.

==================  ===============  ==========================================================================
Check               When             Result when it fails
==================  ===============  ==========================================================================
``architecture``    before download  Blocked: the installed transformers does not know the model type (no
                                     ``trust_remote_code``, ever)
``precision``       before download  Blocked: this device cannot run the checkpoint's published precision
                                     (FP8/FP4 need CUDA), or a precision other than the published one was asked
``memory``          before download  Blocked: weights (safetensors metadata) + KV for the conversation set +
                                     headroom do not fit the budget
``access``          before download  Blocked: a gated repo whose license is not accepted, or no token
``analyzers``       before download  Warning: features Innards does not analyze yet (monitored at the light tier)
``loaded``          after load       Blocked: measured memory after loading is over the budget
==================  ===============  ==========================================================================

::

    from innards.validate import validate
    report = validate("ibm-granite/granite-swash-3b-a600m", device="mps", budget_gb=4)
    report.ok        # False
    print(report)    # "Blocked · memory: needs 7 GB, this machine allows 4 GB ..."
"""

from __future__ import annotations

import json
import logging
import os
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from innards._env import budget_bytes
from innards.features.base import fmt_bytes, text_config

__all__ = ["Check", "Report", "check_loaded", "checkpoint_weights", "published_precision", "validate"]

log = logging.getLogger("innards.validate")

_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F4": 0.5,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
_LOW_PRECISION = {"fp8", "fp4", "mxfp4", "nvfp4", "fbgemm_fp8", "finegrained_fp8"}
HEADROOM_FRACTION = 0.10
HEADROOM_MIN_BYTES = 500_000_000


@dataclass
class Check:
    id: str
    title: str
    status: str
    """``ok``, ``warning``, ``blocked`` or ``skipped`` (could not be checked, e.g. offline)."""
    message: str
    numbers: dict[str, Any] = field(default_factory=dict)
    fix: str | None = None
    when: str = "before download"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Report:
    model: str
    device: str
    checks: list[Check]
    precision: str | None = None
    config: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """``True`` when nothing is blocked (warnings still load)."""
        return not self.blocked

    @property
    def blocked(self) -> list[Check]:
        return [c for c in self.checks if c.status == "blocked"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warning"]

    @property
    def status(self) -> str:
        """``Ready``, ``Loads with limits`` or ``Blocked``: the view's status pill."""
        if self.blocked:
            return "Blocked"
        return "Loads with limits" if self.warnings else "Ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "device": self.device,
            "precision": self.precision,
            "ok": self.ok,
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
        }

    def __str__(self) -> str:
        lines = [f"{self.model} on {self.device}: {self.status}"]
        for check in self.checks:
            lines.append(f"  [{check.status}] {check.title}: {check.message}")
            if check.fix and check.status in {"blocked", "warning"}:
                lines.append(f"      fix: {check.fix}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- facts about a checkpoint


def _local(model: str) -> Path | None:
    path = Path(model).expanduser()
    return path if path.is_dir() else None


def _header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        if size > 100_000_000:
            raise ValueError(f"{path.name}: implausible safetensors header")
        data: dict[str, Any] = json.loads(fh.read(size))
    return data


def _folder_weights(folder: Path) -> tuple[int, dict[str, int]] | None:
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        return None
    total = 0.0
    by_dtype: dict[str, int] = {}
    for file in files:
        for name, info in _header(file).items():
            if name == "__metadata__" or not isinstance(info, dict):
                continue
            n = 1
            for dim in info.get("shape") or []:
                n *= int(dim)
            dtype = str(info.get("dtype", "BF16"))
            by_dtype[dtype] = by_dtype.get(dtype, 0) + n
            total += n * _DTYPE_BYTES.get(dtype, 2)
    return int(total), by_dtype


def checkpoint_weights(
    model: str, *, hub: Any = None, token: str | bool | None = None
) -> tuple[int, dict[str, int]] | None:
    """Bytes of a checkpoint's weights and elements per safetensors dtype, without downloading them.

    Reads a local folder's safetensors headers, or the Hub's safetensors metadata. ``None`` when neither
    is available (offline, or no safetensors).
    """
    folder = _local(model)
    if folder is not None:
        return _folder_weights(folder)
    try:
        if hub is None:
            from huggingface_hub import HfApi

            hub = HfApi(token=token)
        info = hub.model_info(model, expand=["safetensors"])
        meta = getattr(info, "safetensors", None)
        params = dict(getattr(meta, "parameters", None) or {})
        if params:
            total = sum(int(n) * _DTYPE_BYTES.get(str(d), 2) for d, n in params.items())
            return int(total), {str(k): int(v) for k, v in params.items()}
    except Exception as exc:  # offline, gated or unknown repo
        log.debug("no safetensors metadata for %s: %s", model, exc)
    return None


def published_precision(config: Mapping[str, Any], dtypes: Mapping[str, int] | None = None) -> str:
    """The precision the vendor published: ``bf16``, ``fp16``, ``fp32``, ``fp8``, ``fp4/fp8`` ..."""
    quant = config.get("quantization_config")
    method = str(quant.get("quant_method") or "").lower() if isinstance(quant, Mapping) else ""
    expert = str(config.get("expert_dtype") or "").lower()
    if method or expert:
        parts = sorted({p for p in (expert, "fp8" if "fp8" in method else method) if p})
        return "/".join(parts) if parts else "quantized"
    if dtypes:
        if any(d.startswith("F8") for d in dtypes):
            return "fp8"
        if "F4" in dtypes:
            return "fp4"
    cfg = text_config(config)
    raw = str(cfg.get("torch_dtype") or cfg.get("dtype") or config.get("torch_dtype") or config.get("dtype") or "")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(raw.replace("torch.", ""), raw or "bf16")


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------- checks


def _check_architecture(config: Mapping[str, Any]) -> Check:
    model_type = str(config.get("model_type") or "")
    try:
        import transformers
        from transformers import CONFIG_MAPPING
    except ImportError:
        return Check(
            "architecture",
            "Architecture support",
            "blocked",
            "transformers is not installed.",
            fix='pip install "innards[hf]"',
        )
    version = transformers.__version__
    if model_type and model_type in CONFIG_MAPPING:
        remote = " (the config also names remote code, which is not needed)" if config.get("auto_map") else ""
        return Check(
            "architecture",
            "Architecture support",
            "ok",
            f"transformers {version} supports `{model_type}` natively{remote}.",
            {"model_type": model_type, "transformers": version},
        )
    if config.get("auto_map"):
        return Check(
            "architecture",
            "Architecture support",
            "blocked",
            f"transformers {version} doesn't support `{model_type}`; the checkpoint ships its own code, and Innards "
            "never enables trust_remote_code.",
            {"model_type": model_type, "transformers": version},
            "Upgrade transformers, or wait for native support.",
        )
    return Check(
        "architecture",
        "Architecture support",
        "blocked",
        f"transformers {version} doesn't support `{model_type or 'this model type'}`.",
        {"model_type": model_type, "transformers": version},
        "Upgrade transformers, or wait for support.",
    )


def _check_precision(name: str, published: str, requested: str | None, device: str) -> Check:
    kind = device.split(":")[0]
    low = any(p in _LOW_PRECISION for p in published.split("/"))
    if low and kind != "cuda":
        return Check(
            "precision",
            "Device and precision",
            "blocked",
            f"{name} is published in {published.upper()}; that needs a CUDA GPU, and this device is {kind.upper()}.",
            {"published": published, "device": kind},
            "Pick a checkpoint published at 16 bit (bf16 or fp16), or run this one on a CUDA machine.",
        )
    if requested and requested not in {published, "auto"} and not (requested == "fp32" and kind == "cpu"):
        return Check(
            "precision",
            "Device and precision",
            "blocked",
            f"{name} is published in {published.upper()}; Innards runs original checkpoints at their published "
            f"precision, not {requested.upper()}.",
            {"published": published, "requested": requested},
            f"Load it as {published.upper()}" + (" (or FP32 on a CPU)." if not low else "."),
        )
    if low:
        try:
            import torch

            major, minor = torch.cuda.get_device_capability()
            if (major, minor) < (8, 9) and "fp8" in published:
                return Check(
                    "precision",
                    "Device and precision",
                    "warning",
                    f"FP8 matmuls need compute capability 8.9+; this GPU is {major}.{minor}, so transformers "
                    "dequantizes on the fly (slower, more memory).",
                    {"published": published, "capability": f"{major}.{minor}"},
                )
        except Exception:
            pass
    return Check(
        "precision",
        "Device and precision",
        "ok",
        f"Published in {published.upper()}; {kind.upper()} can run it.",
        {"published": published, "device": kind},
    )


def _check_memory(
    name: str,
    config: Mapping[str, Any],
    weights: int | None,
    kv_tokens: int,
    budget: int,
    set_name: str | None,
    dtype_bytes: int,
) -> Check:
    from innards.kv import predict

    if weights is None:
        return Check(
            "memory",
            "Memory fit",
            "skipped",
            "Weight size unknown: no safetensors metadata (offline, or not a safetensors checkpoint). "
            "Memory is checked again after loading.",
            {"budget_bytes": budget},
        )
    dtype = {1: "float8", 2: "bfloat16", 4: "float32"}.get(dtype_bytes, "bfloat16")
    try:
        kv = predict(config, kv_tokens, dtype).total_bytes
    except Exception:
        kv = 0
    headroom = max(int(weights * HEADROOM_FRACTION), HEADROOM_MIN_BYTES)
    needs = weights + kv + headroom
    numbers = {
        "weights_bytes": weights,
        "kv_bytes": kv,
        "kv_tokens": kv_tokens,
        "headroom_bytes": headroom,
        "needs_bytes": needs,
        "budget_bytes": budget,
    }
    what = f"the {set_name} set" if set_name else f"{kv_tokens:,} tokens"
    detail = f"weights {fmt_bytes(weights)} + KV for {what} {fmt_bytes(kv)} + headroom {fmt_bytes(headroom)}"
    if needs > budget:
        return Check(
            "memory",
            "Memory fit",
            "blocked",
            f"needs {needs / 1e9:.0f} GB, this machine allows {budget / 1e9:.0f} GB ({detail}).",
            numbers,
            "Pick a smaller model of the pair, a shorter conversation set, or a machine with more memory.",
        )
    return Check(
        "memory",
        "Memory fit",
        "ok",
        f"needs {needs / 1e9:.1f} GB of the {budget / 1e9:.0f} GB budget ({detail}).",
        numbers,
    )


def _check_access(model: str, hub: Any, token: str | bool | None) -> Check:
    if _local(model) is not None:
        return Check("access", "Access", "ok", "Local folder.")
    try:
        from huggingface_hub import auth_check
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return Check("access", "Access", "skipped", "huggingface_hub is not installed.")
    try:
        if hub is not None and hasattr(hub, "auth_check"):
            hub.auth_check(model)
        else:
            auth_check(model, token=token)
    except GatedRepoError:
        return Check(
            "access",
            "Access",
            "blocked",
            f"{model} is gated and this machine has no access yet.",
            {"repo": model},
            f"Accept the license at https://huggingface.co/{model}, then set HF_TOKEN (or run "
            "`hf auth login`) on this machine.",
        )
    except RepositoryNotFoundError:
        return Check(
            "access",
            "Access",
            "blocked",
            f"No Hub repo named {model} (or it is private).",
            {"repo": model},
            "Check the id, or set HF_TOKEN for a private repo.",
        )
    except Exception as exc:
        return Check("access", "Access", "skipped", f"Could not reach the Hub ({type(exc).__name__}).")
    return Check("access", "Access", "ok", "The repo is public or this machine has access.")


def _check_analyzers(config: Mapping[str, Any]) -> Check:
    from innards.features import feature_status

    status = feature_status(config)
    analyzed = [v["name"] for v in status.values() if v["status"] in {"analyzed", "detected"} and not v.get("light")]
    missing = [v["name"] for v in status.values() if v["status"] == "not_analyzed"]
    if missing:
        return Check(
            "analyzers",
            "Feature analyzers",
            "warning",
            f"The model loads; {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} monitored at the "
            "light tier only (KV, context, memory, timing)."
            + (f" Analyzed in depth: {', '.join(analyzed)}." if analyzed else ""),
            {"analyzed": analyzed, "light_only": missing},
        )
    return Check(
        "analyzers",
        "Feature analyzers",
        "ok",
        f"Analyzed in depth: {', '.join(analyzed)}." if analyzed else "Light tier (no special features).",
        {"analyzed": analyzed},
    )


def validate(
    model: str,
    *,
    precision: str | None = None,
    device: str | None = None,
    conversation_set: Any = None,
    kv_tokens: int | None = None,
    budget_gb: float | None = None,
    config: Mapping[str, Any] | None = None,
    weights_bytes: int | None = None,
    hub: Any = None,
    token: str | bool | None = None,
    check_access: bool = True,
) -> Report:
    """Run every before-download check for ``model`` (a Hub id or a local folder).

    Args:
        precision: The precision asked for (``auto`` = the published one).
        device: ``cuda``, ``mps`` or ``cpu`` (default: the best available here).
        conversation_set: A :class:`~innards.sets.ConversationSet` (or its name); its largest context
            sizes the KV part of the memory check.
        kv_tokens: Context to size the KV for when no set is given (default 8,192).
        budget_gb: Memory budget (default ``$INNARDS_GPU_BUDGET_GB`` or 16 GB).
        config / weights_bytes / hub: Supply facts directly (tests, offline use).
    """
    from innards.kv import load_config
    from innards.lineage.lineup import model_config

    device = device or _default_device()
    budget = budget_bytes(budget_gb)
    name = model.rstrip("/").rsplit("/", 1)[-1]
    checks: list[Check] = []
    if config is None:
        try:
            config = load_config(model, token=token)
        except Exception as exc:
            pinned = model_config(model)
            if pinned is None:
                checks.append(
                    Check(
                        "architecture",
                        "Architecture support",
                        "blocked",
                        f"No config.json for {model}: {type(exc).__name__}: {exc}",
                        {},
                        "Check the Hub id or folder, and the network.",
                    )
                )
                return Report(model, device, checks, precision)
            config = pinned
    config = dict(config)
    checks.append(_check_architecture(config))
    weights = checkpoint_weights(model, hub=hub, token=token) if weights_bytes is None else (weights_bytes, {})
    if weights is None:  # offline: fall back to the lineup's published size, if it is a lineup model
        from innards.lineage.lineup import lineup

        entry = next((m for m in lineup().models.values() if m.id.lower() == model.lower()), None)
        if entry is not None and entry.weights_gb:
            weights = (int(entry.weights_gb * 1e9), {})
    published = published_precision(config, weights[1] if weights else None)
    checks.append(_check_precision(name, published, precision, device))
    set_name = None
    tokens = kv_tokens or 8192
    if conversation_set is not None:
        from innards.sets import count_tokens, get_set

        chosen = get_set(conversation_set) if isinstance(conversation_set, str) else conversation_set
        set_name = chosen.title
        try:
            tokens = count_tokens(chosen)["max_context"]
        except ValueError:
            tokens = kv_tokens or 8192
        from innards.kv import spec_from_config

        try:
            limit = spec_from_config(config).max_context_tokens
        except ValueError:
            limit = None
        if isinstance(limit, int) and tokens > limit:
            checks.append(
                Check(
                    "context",
                    "Context window",
                    "blocked",
                    f"The {set_name} set reaches about {tokens:,} tokens; {name} takes {limit:,}.",
                    {"tokens": tokens, "context_window": limit},
                    "Pick a shorter set or document.",
                )
            )
    dtype_bytes = 1 if published == "fp8" else 2
    if published == "fp32":
        dtype_bytes = 4
    checks.append(_check_memory(name, config, weights[0] if weights else None, tokens, budget, set_name, dtype_bytes))
    if check_access:
        checks.append(_check_access(model, hub, token))
    checks.append(_check_analyzers(config))
    return Report(model, device, checks, precision if precision not in (None, "auto") else published, config)


def check_loaded(model: Any, budget_gb: float | None = None, *, allocated_bytes: int | None = None) -> Check:
    """After loading: measured weights (and device memory in use) against the budget."""
    from innards.collectors.memory import weights_bytes

    budget = budget_bytes(budget_gb)
    weights = weights_bytes(model)
    used = max(weights, allocated_bytes or 0)
    numbers = {"weights_bytes": weights, "allocated_bytes": allocated_bytes, "budget_bytes": budget}
    if used > budget:
        return Check(
            "loaded",
            "Memory after load",
            "blocked",
            f"loaded with {used / 1e9:.1f} GB in use; this machine allows {budget / 1e9:.0f} GB. The model "
            "was unloaded.",
            numbers,
            "Pick a smaller model, or raise INNARDS_GPU_BUDGET_GB if the device has more memory.",
            when="after load",
        )
    return Check(
        "loaded",
        "Memory after load",
        "ok",
        f"{fmt_bytes(used)} in use of the {budget / 1e9:.0f} GB budget.",
        numbers,
        when="after load",
    )


def env_budget() -> float:  # pragma: no cover - convenience for the CLI
    return float(os.environ.get("INNARDS_GPU_BUDGET_GB") or 10)
