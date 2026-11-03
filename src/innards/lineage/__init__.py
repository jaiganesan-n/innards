"""Architecture lineage: what each KV design changed from its predecessor, and why.

One JSON data file per family lives in ``innards/lineage/data``.
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass, field, replace
from importlib import resources
from typing import Any

__all__ = ["LineageEntry", "chain", "explain", "families", "get"]

_ALIASES = {
    "multi-head": "mha",
    "multi_head": "mha",
    "multi-head attention": "mha",
    "multi-query": "mqa",
    "multi_query": "mqa",
    "grouped-query": "gqa",
    "grouped_query": "gqa",
    "latent": "mla",
    "multi-head latent attention": "mla",
    "sliding": "sliding_window",
    "sliding-window": "sliding_window",
    "swa": "sliding_window",
    "local": "sliding_window",
    "local_global": "sliding_window",
    "chunked": "sliding_window",
    "mamba": "ssm",
    "state_space": "ssm",
    "fp8": "fp8_kv",
    "fp8-kv": "fp8_kv",
    "kv_fp8": "fp8_kv",
    "deepseek sparse attention": "dsa",
    "deepseek_sparse_attention": "dsa",
    "sparse": "dsa",
    "csa": "compressed",
    "hca": "compressed",
    "csa_hca": "compressed",
    "csa/hca": "compressed",
    "compressed_attention": "compressed",
}


@dataclass(frozen=True)
class LineageEntry:
    """One step in the architecture lineage."""

    family: str
    name: str
    predecessor: str | None
    change: str
    problem_solved: str
    trade_off: str
    kv_formula: str
    reference: dict[str, Any] = field(default_factory=dict)
    examples: tuple[str, ...] = ()
    model_id: str | None = None
    """Set when :func:`explain` was called with a model."""
    detail: str | None = None
    """Model-specific description of the change, e.g. "8 KV heads shared by 24 query heads"."""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["examples"] = list(self.examples)
        return data

    def summary(self) -> dict[str, str | None]:
        """Compact form stored in ``TurnRecord.model.lineage``."""
        return {"family": self.family, "predecessor": self.predecessor, "change": self.detail or self.change}

    def __str__(self) -> str:
        head = f"{self.name} ({self.family})"
        if self.model_id:
            head = f"{self.model_id}: {head}"
        lines = [head]
        if self.detail:
            lines.append(f"  This model: {self.detail}")
        lines += [
            f"  From: {self.predecessor or '-'}",
            f"  Change: {self.change}",
            f"  Problem solved: {self.problem_solved}",
            f"  Trade-off: {self.trade_off}",
            f"  KV: {self.kv_formula}",
        ]
        if self.reference:
            ref = self.reference
            lines.append(f"  Reference: {ref.get('authors')}, {ref.get('title')} ({ref.get('year')}) {ref.get('url')}")
        return "\n".join(lines)


@functools.cache
def _load() -> dict[str, LineageEntry]:
    entries: dict[str, LineageEntry] = {}
    for item in resources.files(__package__).joinpath("data").iterdir():
        if not item.name.endswith(".json"):
            continue
        raw = json.loads(item.read_text(encoding="utf-8"))
        raw["examples"] = tuple(raw.get("examples", ()))
        entry = LineageEntry(**raw)
        entries[entry.family] = entry
    return entries


def families() -> list[str]:
    """All families with a lineage entry."""
    return sorted(_load())


def _family_key(name: str) -> str | None:
    key = name.strip().lower()
    key = _ALIASES.get(key, key).replace("-", "_")
    return key if key in _load() else None


def get(family: str) -> LineageEntry:
    """Return the entry for a family name or alias (``"gqa"``, ``"sliding"``, ``"fp8"`` ...)."""
    key = _family_key(family)
    if key is None:
        raise KeyError(f"Unknown architecture family {family!r}; known: {', '.join(families())}")
    return _load()[key]


def chain(family: str) -> list[LineageEntry]:
    """The entry for ``family`` followed by its predecessors, oldest last."""
    out = [get(family)]
    while out[-1].predecessor and len(out) < 16:
        out.append(get(out[-1].predecessor))
    return out


def _detail(spec: Any) -> str | None:
    from innards.kv.config import ModelSpec

    if not isinstance(spec, ModelSpec):
        return None
    attn = [layer for layer in spec.layers if layer.has_kv]
    h, hkv = spec.num_attention_heads, spec.num_kv_heads
    parts: list[str] = []
    if spec.attention == "mla":
        parts.append(
            f"caches a {spec.kv_lora_rank}-dim latent + {spec.qk_rope_head_dim}-dim RoPE key per layer"
            + (f" instead of per-head K/V for {h} heads" if h else "")
        )
    elif spec.attention == "gqa" and h and hkv:
        parts.append(f"{hkv} KV heads shared by {h} query heads ({h // hkv} per group)")
    elif spec.attention == "mqa" and h:
        parts.append(f"1 KV head shared by {h} query heads")
    elif spec.attention == "mha" and h:
        parts.append(f"{h} KV heads, one per query head")
    compressed = [layer for layer in attn if layer.is_compressed]
    if compressed:
        csa = sum(1 for layer in compressed if layer.indexer_elements)
        window = {layer.max_tokens for layer in attn if layer.max_tokens is not None}
        parts.append(
            f"{len(attn)} layers keep a {'/'.join(str(w + 1) for w in sorted(window))}-token window; {csa} CSA and "
            f"{len(compressed) - csa} HCA layers add pooled entries"
        )
    indexed = [layer for layer in attn if layer.indexer_elements and not layer.is_compressed]
    if indexed:
        parts.append(f"{len(indexed)} layers also cache a {indexed[0].indexer_elements}-dim indexer key per token")
    local = [layer for layer in attn if layer.is_local and not layer.is_compressed]
    if local:
        caps = "/".join(str(c) for c in sorted({layer.max_tokens for layer in local if layer.max_tokens is not None}))
        glob = len(attn) - len(local) - len(compressed)
        parts.append(
            f"{len(local)} of {spec.num_layers} layers keep the last {caps} tokens"
            + (f"; {glob} global layers grow" if glob else "")
            + ("" if glob or compressed else "; no layer grows past the window")
        )
    recurrent = [layer for layer in spec.layers if layer.is_recurrent]
    if recurrent:
        parts.append(
            f"{len(attn)} of {spec.num_layers} layers use attention; {len(recurrent)} carry fixed recurrent state"
        )
    return "; ".join(parts) or None


def explain(model_or_architecture: Any) -> LineageEntry:
    """Explain what an architecture changed from its predecessor, the problem it solved and its trade-off.

    Args:
        model_or_architecture: A family name or alias (``"mha"``, ``"gqa"``, ``"mla"``,
            ``"sliding_window"``, ``"hybrid"``, ``"fp8"`` ...), a Hub repo id, a config
            path or dict, or a :class:`~innards.kv.ModelSpec`.

    Returns:
        A :class:`LineageEntry`. For a model, ``model_id`` and ``detail`` describe
        that model's own numbers.
    """
    if isinstance(model_or_architecture, str) and _family_key(model_or_architecture):
        return get(model_or_architecture)
    from innards.kv.calculator import resolve_spec

    spec = resolve_spec(model_or_architecture)
    entry = get(spec.architecture)
    return replace(entry, model_id=spec.model_id, detail=_detail(spec))
