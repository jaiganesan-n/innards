"""Architectural KV-cache math: bytes per token, total bytes and capacity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from innards.kv.config import ConfigSource, LayerKV, ModelSpec, load_config, spec_from_config
from innards.kv.dtypes import dtype_bytes, normalize_dtype

__all__ = ["KVPrediction", "layer_bytes", "layer_parts", "predict", "resolve_spec"]


@dataclass(frozen=True)
class KVPrediction:
    """Architectural KV prediction for one model at one context length.

    "Architectural" means what the attention design requires. An engine may cache
    more (for example decompressed MLA heads) or round up to blocks; those
    differences show up as measured-vs-predicted gaps and are never forced away.
    """

    model_id: str | None
    model_type: str | None
    architecture: str
    attention: str | None
    kv_dtype: str
    bytes_per_element: int
    context_tokens: int
    bytes_per_token: int
    """KV growth per token while every attention layer is still growing (windows not yet full)."""
    bytes_per_token_global: int
    """KV growth per token once every local (sliding/chunked) layer is full. Compressed layers
    (DeepSeek-V4 CSA/HCA) contribute their entries amortized per token, rounded to whole bytes."""
    kv_bytes: int
    """Attention KV bytes at ``context_tokens`` (all attention layers)."""
    recurrent_state_bytes: int | None
    """Fixed recurrent state (hybrid/SSM layers); ``None`` if the size is unknown."""
    total_bytes: int
    """``kv_bytes`` plus recurrent state when known."""
    per_layer_bytes: tuple[int, ...]
    by_layer_type: dict[str, int]
    """``{"global": ..., "sliding": ..., "recurrent": ...}``, plus ``compressed`` and ``indexer`` for
    models that have them (DeepSeek-V4 CSA/HCA entries, lightning-indexer keys)."""
    capacity_tokens: int | None
    """Largest context that fits ``memory_bytes`` (and the model's max context)."""
    max_context_tokens: int | None
    memory_bytes: int | None
    batch_size: int
    formula: str
    layers: tuple[LayerKV, ...] = field(repr=False)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["per_layer_bytes"] = list(self.per_layer_bytes)
        data["layers"] = [asdict(layer) for layer in self.layers]
        data["notes"] = list(self.notes)
        return data


def resolve_spec(
    model_or_config: ConfigSource | ModelSpec | Any,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> ModelSpec:
    """Return a :class:`ModelSpec` for a repo id, path, config dict or existing spec."""
    if isinstance(model_or_config, ModelSpec):
        return model_or_config
    model_id = model_or_config if isinstance(model_or_config, str) else None
    config = load_config(model_or_config, revision=revision, token=token)
    return spec_from_config(config, model_id=model_id)


def layer_parts(layer: LayerKV, tokens: int, bytes_per_element: int, batch_size: int = 1) -> dict[str, int]:
    """Bytes one layer holds at ``tokens`` tokens of context, by part.

    ``kv`` (attention K/V; the window for local layers), ``compressed`` (CSA/HCA entries),
    ``indexer`` (lightning-indexer keys) and ``state`` (fixed recurrent state).
    """
    parts = {"kv": 0, "compressed": 0, "indexer": 0, "state": 0}
    if tokens <= 0:
        return parts
    b, batch = bytes_per_element, batch_size
    if layer.has_kv:
        resident = tokens if layer.max_tokens is None else min(tokens, layer.max_tokens)
        parts["kv"] = resident * layer.kv_elements_per_token * b * batch
        if layer.compress_ratio:
            entries = tokens // layer.compress_ratio
            parts["compressed"] = entries * layer.compressed_elements_per_entry * b * batch
            parts["indexer"] = entries * layer.indexer_elements * b * batch
        elif layer.indexer_elements:
            parts["indexer"] = tokens * layer.indexer_elements * b * batch
    if layer.is_recurrent and layer.state_elements:
        fp32 = min(layer.state_fp32_elements, layer.state_elements)
        parts["state"] = ((layer.state_elements - fp32) * b + fp32 * 4) * batch
    return parts


def layer_bytes(layer: LayerKV, tokens: int, bytes_per_element: int, batch_size: int = 1) -> tuple[int, int]:
    """Return ``(kv_bytes, state_bytes)`` for one layer holding ``tokens`` tokens of context.

    ``kv_bytes`` counts everything that grows with context: K/V, compressed entries and indexer keys.
    """
    parts = layer_parts(layer, tokens, bytes_per_element, batch_size)
    return parts["kv"] + parts["compressed"] + parts["indexer"], parts["state"]


def _totals(spec: ModelSpec, tokens: int, b: int, batch: int) -> tuple[int, int | None, list[int], dict[str, int]]:
    per_layer: list[int] = []
    groups = {"global": 0, "sliding": 0, "recurrent": 0}
    if any(layer.is_compressed for layer in spec.layers):
        groups["compressed"] = 0
    if any(layer.indexer_elements for layer in spec.layers):
        groups["indexer"] = 0
    kv_total = 0
    state_total = 0
    state_known = True
    for layer in spec.layers:
        parts = layer_parts(layer, tokens, b, batch)
        if layer.is_recurrent and layer.state_elements is None and tokens > 0:
            state_known = False
        grow = parts["kv"] + parts["compressed"] + parts["indexer"]
        per_layer.append(grow + parts["state"])
        kv_total += grow
        state_total += parts["state"]
        if parts["kv"]:
            groups["sliding" if layer.is_local else "global"] += parts["kv"]
        if parts["compressed"]:
            groups["compressed"] += parts["compressed"]
        if parts["indexer"]:
            groups["indexer"] += parts["indexer"]
        groups["recurrent"] += parts["state"]
    return kv_total, (state_total if state_known else None), per_layer, groups


def _per_token_extra(layer: LayerKV) -> int:
    """Indexer elements added per token by a DSA layer (compressed layers add theirs per entry)."""
    return 0 if layer.compress_ratio else layer.indexer_elements


def _fmt(n: int) -> str:
    return f"{n:,}"


def _formula(spec: ModelSpec, b: int, bytes_per_token: int) -> str:
    attn = [layer for layer in spec.layers if layer.has_kv]
    if not attn:
        return "No attention layers: context adds no KV; only fixed recurrent state."
    compressed = [layer for layer in attn if layer.is_compressed]
    if compressed:
        window = {layer.max_tokens for layer in attn if layer.max_tokens is not None}
        elems = attn[0].kv_elements_per_token
        text = (
            f"{len(attn)} layers × {elems} shared K=V × {b} B = {_fmt(len(attn) * elems * b)} B/token "
            f"until the {'/'.join(str(w + 1) for w in sorted(window))}-token windows fill"
        )
        by_ratio: dict[int, list[LayerKV]] = {}
        for layer in compressed:
            by_ratio.setdefault(int(layer.compress_ratio or 1), []).append(layer)
        for ratio, group in sorted(by_ratio.items()):
            per_entry = group[0].compressed_elements_per_entry + group[0].indexer_elements
            idx = f" ({group[0].indexer_elements} of them indexer keys)" if group[0].indexer_elements else ""
            kind = "CSA" if group[0].indexer_elements else "HCA"
            text += f"; {len(group)} {kind} layers add {per_entry} elements{idx} per {ratio} tokens"
        return text
    if spec.attention == "mla":
        rank, rope = spec.kv_lora_rank or 0, spec.qk_rope_head_dim or 0
        head = f"{len(attn)} layers × ({rank} latent + {rope} RoPE"
        indexer = attn[0].indexer_elements
        head += f" + {indexer} indexer key) × {b} B" if indexer else f") × {b} B"
    else:
        heads = sorted({layer.kv_elements_per_token // (2 * (spec.head_dim or 1)) for layer in attn})
        heads_txt = "/".join(map(str, heads))
        head = f"2 × {len(attn)} layers × {heads_txt} KV heads × {spec.head_dim} head dim × {b} B"
    text = f"{head} = {_fmt(bytes_per_token)} B/token"
    local = [layer for layer in attn if layer.is_local]
    if local:
        caps = "/".join(str(c) for c in sorted({layer.max_tokens for layer in local if layer.max_tokens is not None}))
        n_global = len(attn) - len(local)
        if n_global:
            text += f"; {len(local)} local layers stop at {caps} tokens, then only {n_global} global layers grow"
        else:
            text += f"; every layer is local and stops at {caps} tokens"
    return text


def _capacity(spec: ModelSpec, memory_bytes: int, b: int, batch: int) -> int:
    limit = spec.max_context_tokens or 2**40

    def fits(n: int) -> bool:
        kv, state, _, _ = _totals(spec, n, b, batch)
        return kv + (state or 0) <= memory_bytes

    if fits(limit):
        return limit
    lo, hi = 0, limit  # fits(lo) is True for lo == 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def predict(
    model_or_config: ConfigSource | ModelSpec | Any,
    context_tokens: int,
    kv_dtype: str = "float16",
    *,
    memory_bytes: int | None = None,
    batch_size: int = 1,
    revision: str | None = None,
    token: str | bool | None = None,
) -> KVPrediction:
    """Predict architectural KV-cache size from a model config, without loading weights.

    Args:
        model_or_config: Hub repo id (only ``config.json`` is fetched), a path to a
            config file or model directory, a config dict, or a ``PretrainedConfig``.
        context_tokens: Tokens resident in the cache.
        kv_dtype: Cache element type: ``float32``, ``float16``, ``bfloat16`` or
            ``float8`` (aliases ``fp32``, ``fp16``, ``bf16``, ``fp8``).
        memory_bytes: Optional KV memory budget used for ``capacity_tokens``.
        batch_size: Sequences held in the cache at once.
        revision: Hub revision for repo ids.
        token: Hub token for gated repos.

    Returns:
        A :class:`KVPrediction`.
    """
    if context_tokens < 0:
        raise ValueError("context_tokens must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    spec = resolve_spec(model_or_config, revision=revision, token=token)
    dtype = normalize_dtype(kv_dtype)
    b = dtype_bytes(dtype)

    kv, state, per_layer, groups = _totals(spec, context_tokens, b, batch_size)
    attn = [layer for layer in spec.layers if layer.has_kv]
    rate = sum(layer.kv_elements_per_token + _per_token_extra(layer) for layer in attn) * b * batch_size
    rate_global = round(
        sum(
            (0 if layer.is_local else layer.kv_elements_per_token + layer.indexer_elements)
            + (
                (layer.compressed_elements_per_entry + layer.indexer_elements) / layer.compress_ratio
                if layer.compress_ratio
                else 0
            )
            for layer in attn
        )
        * b
        * batch_size
    )
    notes = spec.notes
    if spec.max_context_tokens is not None and context_tokens > spec.max_context_tokens:
        notes = (*notes, f"context_tokens exceeds the model's max context ({spec.max_context_tokens:,}).")
    capacity = spec.max_context_tokens
    if memory_bytes is not None:
        capacity = _capacity(spec, memory_bytes, b, batch_size)

    return KVPrediction(
        model_id=spec.model_id,
        model_type=spec.model_type,
        architecture=spec.architecture,
        attention=spec.attention,
        kv_dtype=dtype,
        bytes_per_element=b,
        context_tokens=context_tokens,
        bytes_per_token=rate,
        bytes_per_token_global=rate_global,
        kv_bytes=kv,
        recurrent_state_bytes=state,
        total_bytes=kv + (state or 0),
        per_layer_bytes=tuple(per_layer),
        by_layer_type=groups,
        capacity_tokens=capacity,
        max_context_tokens=spec.max_context_tokens,
        memory_bytes=memory_bytes,
        batch_size=batch_size,
        formula=_formula(spec, b, rate // batch_size),
        layers=spec.layers,
        notes=notes,
    )
