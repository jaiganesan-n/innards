"""Read KV-cache sizes from a transformers cache object: shapes and dtypes only.

The installed transformers version decides the cache layout, so nothing here
assumes attribute names up front. :func:`snapshot_cache` discovers the layout
once per cache/layer class and then copies tensor metadata (shape, bytes per
element). It never copies tensors, never moves them between devices and never
reads their values, so it cannot force a device synchronization. A view that aliases
memory already counted (same address, shape and dtype, e.g. ``values is keys``) is counted
once, and every per-token tensor a layer keeps is counted, not only ``keys`` and ``values``.

:func:`measure_snapshot` turns a snapshot into bytes (``numel x element_size``)
and runs in an observer worker, off the generation path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["CacheSnapshot", "KVMeasurement", "LayerSnapshot", "measure_snapshot", "snapshot_cache"]

TensorMeta = tuple[tuple[int, ...], int]
"""``(shape, bytes_per_element)``."""

# Attribute names that hold bookkeeping (counters, dtype/device handles), not cached states.
_IGNORED = {
    "_sliding_window_tensor",
    "cumulative_length",
    "indexer_cumulative_length",
    "idx_cumulative_length",
    "_seen_tokens",
    "cache_position",
    "dtype",
    "device",
    "indexer_dtype",
    "indexer_device",
}
_RECURRENT_HINTS = ("conv", "recurrent", "ssm", "state")

ViewKey = tuple[int, tuple[int, ...], int]
"""``(data_ptr, shape, bytes_per_element)``: identical views of the same memory."""


@dataclass(frozen=True, slots=True)
class LayerSnapshot:
    index: int
    kind: str
    """``global``, ``sliding``, ``recurrent``, ``hybrid`` or ``hybrid_sliding``."""
    kv: tuple[TensorMeta, ...]
    state: tuple[TensorMeta, ...]
    extra: tuple[tuple[str, TensorMeta], ...] = ()
    """Per-token tensors reported in their own group: ``compressed`` (CSA/HCA entries), ``indexer``
    (lightning-indexer keys) and ``pending`` (engine buffers of tokens waiting to fill a window)."""


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    layout: str
    """How the cache was read: ``layers`` (transformers Cache API), ``key_value_lists``,
    ``legacy_tuples`` or ``tensor_walk``."""
    layers: tuple[LayerSnapshot, ...]
    cache_class: str


@dataclass(frozen=True, slots=True)
class KVMeasurement:
    total_bytes: int
    per_layer_bytes: tuple[int, ...]
    by_layer_type: dict[str, int]
    layer_tokens: tuple[int | None, ...]
    """Resident tokens per layer, read from the sequence dimension of K (``None`` for state-only layers)."""
    layout: str
    source: str = "hf_cache"


def _is_tensor(obj: Any) -> bool:
    # Duck-typed so this module never imports torch.
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "element_size")


def _meta(tensor: Any) -> TensorMeta:
    itemsize = getattr(tensor.dtype, "itemsize", None)
    if itemsize is None:  # torch < 2.1
        itemsize = tensor.element_size()
    return tuple(int(d) for d in tensor.shape), int(itemsize)


def _view_key(tensor: Any) -> ViewKey | None:
    """Identity of a tensor view: the same address, shape and element size is the same memory.

    ``data_ptr()`` is a host-side pointer (no device synchronization). Empty and meta tensors
    report 0 and are never deduplicated; they hold no bytes anyway.
    """
    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr):
        return None
    try:
        address = int(data_ptr())
    except Exception:
        return None
    if address == 0:
        return None
    shape, itemsize = _meta(tensor)
    return address, shape, itemsize


def _unique(tensors: list[Any], seen: set[ViewKey]) -> list[Any]:
    """Drop tensors that alias a view already counted (e.g. DeepSeek-V4 keeps ``values is keys``)."""
    out = []
    for tensor in tensors:
        key = _view_key(tensor)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out.append(tensor)
    return out


def _tensors_in(value: Any) -> list[Any]:
    """Cached tensors held directly in an attribute (tensor, list/tuple or dict of tensors)."""
    return [t for _, t in _named_tensors_in(value, "")]


def _named_tensors_in(value: Any, path: str) -> list[tuple[str, Any]]:
    """``(path, tensor)`` for cached tensors in an attribute; dict keys extend the path."""
    if value is None:
        return []
    if _is_tensor(value):
        return [(path, value)] if len(value.shape) >= 2 else []  # skip scalars / bookkeeping vectors
    if isinstance(value, Mapping):
        return [item for k, v in value.items() for item in _named_tensors_in(v, f"{path}.{k}")]
    if isinstance(value, (list, tuple)):
        return [item for v in value for item in _named_tensors_in(v, path)]
    return []


def _extra_group(path: str) -> str | None:
    """Group for a per-token tensor that is not plain K/V, or ``None`` for K/V.

    ``pending`` (tokens waiting to fill a compression window, and the overlap carried to the next
    one), ``indexer`` (lightning-indexer keys, compressed or not) and ``compressed`` (CSA/HCA entries).
    """
    lowered = path.lower()
    if any(word in lowered for word in ("buffer", "overlap")):
        return "pending"
    if "index" in lowered:
        return "indexer"
    if "compress" in lowered:
        return "compressed"
    return None


# Per layer class: (kv attribute names, state attribute names). Discovered once.
_LAYER_PLANS: dict[type, tuple[tuple[str, ...], tuple[str, ...]]] = {}


def _plan_for(layer: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cls = type(layer)
    plan = _LAYER_PLANS.get(cls)
    if plan is None:
        kv: list[str] = []
        state: list[str] = []
        attrs = dict(vars(layer)) if hasattr(layer, "__dict__") else {}
        for slot in getattr(cls, "__slots__", ()):
            if hasattr(layer, slot):
                attrs.setdefault(slot, getattr(layer, slot))
        for name, value in attrs.items():
            if name in _IGNORED:
                continue
            if _is_tensor(value) or isinstance(value, (Mapping, list, tuple)) or value is None:
                lowered = name.lower()
                if any(hint in lowered for hint in _RECURRENT_HINTS):
                    state.append(name)
                else:
                    # Keys and values, plus every other per-token tensor the layer keeps
                    # (indexer keys, compressed entries and their pending buffers).
                    kv.append(name)
        # K and V first, so the sequence length is read from K.
        kv.sort(key=lambda n: (n not in {"keys", "values"}, "key" not in n.lower() and "value" not in n.lower()))
        plan = (tuple(kv), tuple(state))
        # Only memoize once something was found; lazily-initialized layers may be empty at first.
        if kv or state:
            _LAYER_PLANS[cls] = plan
    return plan


def _layer_kind(layer: Any, has_kv: bool, has_state: bool) -> str:
    sliding = bool(getattr(layer, "is_sliding", False))
    if has_state and has_kv:
        return "hybrid_sliding" if sliding else "hybrid"
    if has_state:
        return "recurrent"
    return "sliding" if sliding else "global"


def _snapshot_layers_api(cache: Any) -> CacheSnapshot:
    out: list[LayerSnapshot] = []
    seen: set[ViewKey] = set()
    for i, layer in enumerate(cache.layers):
        kv_names, state_names = _plan_for(layer)
        named = [item for name in kv_names for item in _named_tensors_in(getattr(layer, name, None), name)]
        kept = {id(t) for t in _unique([t for _, t in named], seen)}
        kv_list: list[TensorMeta] = []
        extra: list[tuple[str, TensorMeta]] = []
        for path, tensor in named:
            if id(tensor) not in kept:
                continue
            kept.discard(id(tensor))  # the same tensor object listed twice is counted once
            group = _extra_group(path)
            if group is None:
                kv_list.append(_meta(tensor))
            else:
                extra.append((group, _meta(tensor)))
        state_tensors = [t for name in state_names for t in _tensors_in(getattr(layer, name, None))]
        state = tuple(_meta(t) for t in _unique(state_tensors, seen))
        kv = tuple(kv_list)
        has_state = bool(state_names) and any(_tensors_in(getattr(layer, n, None)) for n in state_names)
        kind = _layer_kind(layer, bool(kv), has_state or (bool(state_names) and not kv))
        out.append(LayerSnapshot(i, kind, kv, state, tuple(extra)))
    return CacheSnapshot("layers", tuple(out), type(cache).__name__)


def _snapshot_lists(cache: Any) -> CacheSnapshot:
    keys = list(getattr(cache, "key_cache", []) or [])
    values = list(getattr(cache, "value_cache", []) or [])
    conv = list(getattr(cache, "conv_states", []) or [])
    ssm = list(getattr(cache, "ssm_states", []) or [])
    sliding = getattr(cache, "is_sliding", None)
    n = max(len(keys), len(values), len(conv), len(ssm))
    out: list[LayerSnapshot] = []
    seen: set[ViewKey] = set()
    for i in range(n):
        kv_tensors = [t for seq in (keys, values) if i < len(seq) for t in _tensors_in(seq[i])]
        state_tensors = [t for seq in (conv, ssm) if i < len(seq) for t in _tensors_in(seq[i])]
        kv = tuple(_meta(t) for t in _unique(kv_tensors, seen))
        state = tuple(_meta(t) for t in _unique(state_tensors, seen))
        # Hybrid caches keep empty placeholder tensors for layers that do not use them.
        kv = tuple(m for m in kv if math.prod(m[0]) > 0)
        state = tuple(m for m in state if math.prod(m[0]) > 0)
        is_sliding = bool(sliding[i]) if isinstance(sliding, (list, tuple)) and i < len(sliding) else False
        kind = "hybrid" if kv and state else ("recurrent" if state else ("sliding" if is_sliding else "global"))
        out.append(LayerSnapshot(i, kind, kv, state))
    return CacheSnapshot("key_value_lists", tuple(out), type(cache).__name__)


def _snapshot_tuples(cache: Iterable[Any]) -> CacheSnapshot:
    out = [LayerSnapshot(i, "global", tuple(_meta(t) for t in _tensors_in(layer)), ()) for i, layer in enumerate(cache)]
    return CacheSnapshot("legacy_tuples", tuple(out), type(cache).__name__)


def _snapshot_walk(cache: Any) -> CacheSnapshot:
    tensors = [t for value in vars(cache).values() for t in _tensors_in(value)]
    layer = LayerSnapshot(0, "global", tuple(_meta(t) for t in tensors), ())
    return CacheSnapshot("tensor_walk", (layer,), type(cache).__name__)


def snapshot_cache(cache: Any) -> CacheSnapshot | None:
    """Copy tensor metadata out of a cache object. Metadata only: safe on the generation thread."""
    if cache is None:
        return None
    if isinstance(getattr(cache, "layers", None), (list, tuple)):
        return _snapshot_layers_api(cache)
    if hasattr(cache, "key_cache") or hasattr(cache, "conv_states"):
        return _snapshot_lists(cache)
    if isinstance(cache, (list, tuple)):
        return _snapshot_tuples(cache)
    if hasattr(cache, "__dict__"):
        return _snapshot_walk(cache)
    return None


def _nbytes(meta: TensorMeta) -> int:
    shape, itemsize = meta
    return math.prod(shape) * itemsize


def measure_snapshot(snapshot: CacheSnapshot) -> KVMeasurement:
    """Bytes per layer and per layer type from a snapshot (``numel x element_size``)."""
    per_layer: list[int] = []
    tokens: list[int | None] = []
    groups = {"global": 0, "sliding": 0, "recurrent": 0}
    for layer in snapshot.layers:
        kv = sum(_nbytes(m) for m in layer.kv)
        state = sum(_nbytes(m) for m in layer.state)
        extra = 0
        for group, meta in layer.extra:
            nbytes = _nbytes(meta)
            groups[group] = groups.get(group, 0) + nbytes
            extra += nbytes
        per_layer.append(kv + state + extra)
        groups["sliding" if "sliding" in layer.kind else "global"] += kv
        groups["recurrent"] += state
        # K is stored as [batch, heads, seq, dim]; read seq from the first KV tensor.
        tokens.append(layer.kv[0][0][-2] if layer.kv and len(layer.kv[0][0]) >= 3 else None)
    return KVMeasurement(
        total_bytes=sum(per_layer),
        per_layer_bytes=tuple(per_layer),
        by_layer_type=groups,
        layer_tokens=tuple(tokens),
        layout=snapshot.layout,
    )
