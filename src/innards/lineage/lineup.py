"""The lineup: pairs of a newer model and the version before it, and what changed between them.

Each pair isolates a new architectural feature (Gated DeltaNet, MoE with a shared expert). Every model
runs on a local machine with 16 GB at 16-bit precision. Pinned config
snapshots ship with Innards, so the view can show a pair's architecture, what changed and the KV
predicted from the config before either model has been run (labeled as predicted).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

__all__ = ["LineupModel", "LineupPair", "lineup", "model_config", "pair_of", "predicted_view", "what_changed"]

_CONTEXTS = (4096, 32768, 131072)


@dataclass(frozen=True)
class LineupPair:
    id: str
    feature: str
    features: tuple[str, ...]
    newer: str
    previous: str
    where: str
    """``local``: runs on a 16 GB machine at 16-bit precision."""
    change: str
    alternates: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "feature": self.feature,
            "features": list(self.features),
            "newer": self.newer,
            "previous": self.previous,
            "where": self.where,
            "change": self.change,
            "alternates": [{"newer": n, "previous": p} for n, p in self.alternates],
        }


@dataclass(frozen=True)
class LineupModel:
    id: str
    config_file: str
    vendor: str
    weights_gb: float | None
    precision: str
    params: int | None = None
    pair: str | None = None
    role: str | None = None
    """``newer`` or ``previous``."""
    partner: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> dict[str, Any]:
        return model_config(self.id) or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vendor": self.vendor,
            "weights_gb": self.weights_gb,
            "precision": self.precision,
            "params": self.params,
            "pair": self.pair,
            "role": self.role,
            "partner": self.partner,
        }


@dataclass(frozen=True)
class Lineup:
    pairs: tuple[LineupPair, ...]
    models: dict[str, LineupModel]

    def to_dict(self) -> dict[str, Any]:
        return {"pairs": [p.to_dict() for p in self.pairs], "models": [m.to_dict() for m in self.models.values()]}


@functools.cache
def lineup() -> Lineup:
    raw = json.loads(resources.files(__package__).joinpath("lineup.json").read_text(encoding="utf-8"))
    pairs = tuple(
        LineupPair(
            id=p["id"],
            feature=p["feature"],
            features=tuple(p.get("features") or ()),
            newer=p["newer"],
            previous=p["previous"],
            where=p.get("where", "local"),
            change=p.get("change", ""),
            alternates=tuple((a["newer"], a["previous"]) for a in p.get("alternates") or ()),
        )
        for p in raw["pairs"]
    )
    roles: dict[str, tuple[str, str, str]] = {}
    for pair in pairs:
        for newer, previous in ((pair.newer, pair.previous), *pair.alternates):
            roles[newer] = (pair.id, "newer", previous)
            roles[previous] = (pair.id, "previous", newer)
    models = {}
    for model_id, info in raw["models"].items():
        pair_id, role, partner = roles.get(model_id, (None, None, None))
        models[model_id] = LineupModel(
            id=model_id,
            config_file=info["config"],
            vendor=info.get("vendor", ""),
            weights_gb=info.get("weights_gb"),
            precision=info.get("precision", "bf16"),
            params=info.get("params"),
            pair=pair_id,
            role=role,
            partner=partner,
        )
    return Lineup(pairs, models)


def _find(model_id: str) -> LineupModel | None:
    models = lineup().models
    if model_id in models:
        return models[model_id]
    low = model_id.lower().rstrip("/")
    for key, model in models.items():
        if key.lower() == low or key.rsplit("/", 1)[-1].lower() == low.rsplit("/", 1)[-1]:
            return model
    return None


@functools.cache
def _config_text(name: str) -> str:
    return resources.files(__package__).joinpath("configs", name).read_text(encoding="utf-8")


def model_config(model_id: str) -> dict[str, Any] | None:
    """The pinned ``config.json`` of a lineup model (``None`` for models outside the lineup)."""
    model = _find(model_id)
    if model is None:
        return None
    data: dict[str, Any] = json.loads(_config_text(model.config_file))
    return data


def pair_of(model_id: str) -> tuple[LineupPair, str, str] | None:
    """``(pair, role, partner id)`` for a lineup model: role ``newer`` or ``previous``."""
    model = _find(model_id)
    if model is None or model.pair is None or model.role is None or model.partner is None:
        return None
    pair = next(p for p in lineup().pairs if p.id == model.pair)
    return pair, model.role, model.partner


def _facts(config: Mapping[str, Any]) -> dict[str, Any]:
    from innards.experts.routing import routing_facts
    from innards.features import detect, feature_status
    from innards.features.base import text_config
    from innards.kv import predict, spec_from_config

    spec = spec_from_config(config)
    cfg = text_config(config)
    routing = routing_facts(cfg)
    shared = routing.get("shared_experts")
    if (
        shared is None
        and routing.get("experts")
        and any(
            cfg.get(k)
            for k in (
                "shared_intermediate_size",
                "shared_expert_intermediate_size",
                "moe_shared_expert_intermediate_size",
            )
        )
    ):
        shared = 1  # sized by an intermediate size rather than counted
    kinds: dict[str, int] = {}
    for layer in spec.layers:
        kinds[layer.layer_type] = kinds.get(layer.layer_type, 0) + 1
    status = feature_status(config, active=[a.name for a, _ in detect(config)])
    features = sorted(k for k, v in status.items() if v["status"] != "absent")
    if cfg.get("moe_latent_size"):
        features.append("latent_moe")
    if shared:
        features.append("shared_expert")
    return {
        "architecture": spec.architecture,
        "attention": spec.attention,
        "layers": spec.num_layers,
        "layer_kinds": kinds,
        "kv_heads": spec.num_kv_heads,
        "heads": spec.num_attention_heads,
        "head_dim": spec.head_dim,
        "hidden_size": cfg.get("hidden_size"),
        "experts": routing.get("experts"),
        "experts_per_token": routing.get("top_k"),
        "shared_experts": shared,
        "context_window": spec.max_context_tokens,
        "features": features,
        "latent_size": cfg.get("moe_latent_size"),
        "kv_per_token_bf16": predict(spec, 1, "bf16").bytes_per_token,
        "kv_at_32k_bf16": predict(spec, 32768, "bf16").total_bytes,
    }


def what_changed(newer: str | Mapping[str, Any], previous: str | Mapping[str, Any]) -> dict[str, Any]:
    """Features added and dropped, and the architecture facts side by side, from two configs."""
    from innards.features import feature_status

    def config_of(x: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(x, Mapping):
            return dict(x)
        found = model_config(x)
        if found is None:
            from innards.kv import load_config

            found = load_config(x)
        return found

    a, b = config_of(newer), config_of(previous)
    fa, fb = _facts(a), _facts(b)
    names: dict[str, Any] = {**feature_status(b), **feature_status(a)}
    names["latent_moe"] = {"name": "LatentMoE (experts work in a smaller latent space)"}
    names["shared_expert"] = {"name": "Shared expert (runs for every token)"}
    rows = []
    labels = {
        "architecture": "Architecture",
        "attention": "Attention heads",
        "layers": "Layers",
        "layer_kinds": "Layer kinds",
        "kv_heads": "KV heads",
        "heads": "Query heads",
        "head_dim": "Head dim",
        "hidden_size": "Hidden size",
        "experts": "Routed experts",
        "experts_per_token": "Experts per token",
        "shared_experts": "Shared experts",
        "latent_size": "Expert latent size",
        "context_window": "Context window",
        "kv_per_token_bf16": "KV per token (bf16, windows filling)",
        "kv_at_32k_bf16": "KV at 32K tokens (bf16)",
    }
    for key, label in labels.items():
        if fa.get(key) != fb.get(key):
            rows.append({"fact": label, "key": key, "previous": fb.get(key), "newer": fa.get(key)})
    added = [f for f in fa["features"] if f not in fb["features"]]
    dropped = [f for f in fb["features"] if f not in fa["features"]]
    result: dict[str, Any] = {
        "added": [{"id": f, "name": names.get(f, {}).get("name", f)} for f in added],
        "dropped": [{"id": f, "name": names.get(f, {}).get("name", f)} for f in dropped],
        "facts": rows,
        "newer": fa,
        "previous": fb,
    }
    if isinstance(newer, str):
        found = pair_of(newer)
        if found is not None:
            result["pair"] = found[0].to_dict()
    return result


def predicted_view(model_id: str, contexts: tuple[int, ...] = _CONTEXTS) -> dict[str, Any]:
    """KV predicted from the config at a few context lengths, for a model that has not run yet."""
    from innards.kv import predict

    config = model_config(model_id)
    if config is None:
        from innards.kv import load_config

        config = load_config(model_id)
    points = []
    for tokens in contexts:
        p = predict(config, tokens, "bf16")
        points.append({"tokens": tokens, "total_bytes": p.total_bytes, "by_layer_type": dict(p.by_layer_type)})
    p = predict(config, contexts[0], "bf16")
    return {
        "model": model_id,
        "label": "predicted",
        "kv_dtype": "bfloat16",
        "formula": p.formula,
        "architecture": p.architecture,
        "points": points,
        "notes": list(p.notes),
    }
