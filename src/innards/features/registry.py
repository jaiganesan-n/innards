"""The feature-analyzer registry and the catalog of features Innards can detect.

``detect(config)`` returns every registered analyzer whose feature the model has. ``feature_status``
also lists features Innards recognizes but does not analyze yet, so the view can mark them
"not analyzed yet" rather than hide them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from innards.features.base import Analyzer, Detection, Payload, TurnData, text_config

__all__ = [
    "KNOWN_FEATURES",
    "FeatureInfo",
    "analyzers",
    "detect",
    "explicit_call",
    "feature_status",
    "get",
    "register",
]

log = logging.getLogger("innards.features")

_REGISTRY: dict[str, Analyzer] = {}


def register(analyzer: Analyzer) -> Analyzer:
    """Add (or replace) an analyzer. New architectures get a new analyzer registered here."""
    if not analyzer.name:
        raise ValueError("An analyzer needs a name")
    _REGISTRY[analyzer.name] = analyzer
    return analyzer


def analyzers() -> list[Analyzer]:
    """Registered analyzers, light ones (no hooks) first."""
    return sorted(_REGISTRY.values(), key=lambda a: (not a.light, a.name))


def get(name: str) -> Analyzer:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"No analyzer named {name!r}; registered: {', '.join(sorted(_REGISTRY))}") from None


def detect(config: Mapping[str, Any], only: Iterable[str] | None = None) -> list[tuple[Analyzer, Detection]]:
    """Analyzers whose feature ``config`` has, with where it lives. ``only`` limits the names."""
    wanted = set(only) if only is not None else None
    out = []
    for analyzer in analyzers():
        if wanted is not None and analyzer.name not in wanted:
            continue
        try:
            found = analyzer.detect(config)
        except Exception:
            log.warning("innards: %s could not read this config", analyzer.name, exc_info=True)
            continue
        if found is not None:
            out.append((analyzer, found))
    return out


# --------------------------------------------------------------------------- features without an analyzer yet


@dataclass(frozen=True)
class FeatureInfo:
    id: str
    name: str
    detect: Callable[[Mapping[str, Any]], bool]
    note: str = ""
    runs: bool = True
    """``False`` for parts of the checkpoint that transformers does not run during generation."""


def _pattern(cfg: Mapping[str, Any]) -> list[str]:
    for key in ("layers_block_type", "layer_types"):
        value = cfg.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
    pattern = cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str):
        return [{"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}.get(c, c) for c in pattern]
    return []


def _has_mamba(cfg: Mapping[str, Any]) -> bool:
    if cfg.get("mamba_num_heads") or cfg.get("mamba_n_heads"):
        return True
    return str(cfg.get("model_type") or "") in {"mamba", "mamba2", "jamba", "bamba", "falcon_h1"}


KNOWN_FEATURES: tuple[FeatureInfo, ...] = (
    FeatureInfo("mamba2", "Mamba-2 state-space layers", _has_mamba, "Fixed state size is shown by the KV analyzer."),
    FeatureInfo(
        "short_conv",
        "Short convolution layers",
        lambda c: str(c.get("model_type") or "") == "lfm2",
        "Their fixed state is shown by the KV analyzer.",
    ),
    FeatureInfo(
        "mtp",
        "Multi-token prediction head",
        lambda c: bool(c.get("num_nextn_predict_layers") or c.get("mtp_num_hidden_layers")),
        "transformers does not run the MTP block during generation, so there is nothing to measure.",
        runs=False,
    ),
)


def feature_status(
    config: Mapping[str, Any], active: Sequence[str] = (), *, all_analyzers: bool = True
) -> dict[str, dict[str, Any]]:
    """Every feature Innards knows, and what it does with it for this model.

    Status per feature: ``analyzed`` (probes attached), ``detected`` (the model has it and an analyzer
    exists, but it is switched off), ``not_analyzed`` (the model has it; Innards has no analyzer yet),
    ``not_run`` (in the checkpoint, but transformers does not run it) or ``absent``.
    """
    cfg = text_config(config)
    out: dict[str, dict[str, Any]] = {}
    found = {a.name: d for a, d in detect(config)}
    for analyzer in analyzers() if all_analyzers else []:
        detection = found.get(analyzer.name)
        if detection is None:
            status = "absent"
        elif analyzer.name in active:
            status = "analyzed"
        else:
            status = "detected"
        out[analyzer.name] = {
            "name": analyzer.title,
            "status": status,
            "light": analyzer.light,
            "layers": len(detection.layers) if detection else 0,
            "reads": analyzer.reads,
            "shows": analyzer.shows,
        }
    for info in KNOWN_FEATURES:
        try:
            present = bool(info.detect(cfg))
        except Exception:
            present = False
        if present:
            out[info.id] = {
                "name": info.name,
                "status": "not_analyzed" if info.runs else "not_run",
                "light": False,
                "note": info.note,
            }
    return out


# --------------------------------------------------------------------------- explicit form


def _captured_turns(args: Any) -> list[Any]:
    if args is None:
        raise ValueError("Pass args=watch.last_turn() (what observe captured), not generate()'s output")
    if isinstance(args, (list, tuple)):
        return list(args)
    return [args]


def explicit_call(analyzer: Analyzer, model: Any, args: Any) -> Payload | None:
    """``analyzer(model=..., args=watch.last_turn())``: analyze turns that ``observe`` captured."""
    payloads = []
    for captured in _captured_turns(args):
        data = getattr(captured, "data", None)
        if not isinstance(data, Mapping):
            raise TypeError(
                "args must be what observe captured (watch.last_turn()), not the output of generate(): "
                "generate() returns token ids, which hold no hook captures"
            )
        turn: TurnData | None = data.get(analyzer.name)
        if turn is None:
            detection = None
            if model is not None:
                from innards.kv.config import load_config

                config = model if isinstance(model, Mapping) else load_config(model)
                detection = analyzer.detect(config)
            if detection is None and not analyzer.light:
                raise ValueError(
                    f"{analyzer.title} was not captured on this turn: the model does not have it, or observe() "
                    f"ran with features that excluded {analyzer.name!r}"
                )
            if detection is None:
                return None
            turn = captured.turn_data(analyzer.name, detection)
        payloads.append(analyzer.analyze(turn))
    return payloads[0] if len(payloads) == 1 else {"turns": payloads}
