"""The Analysis tab, built from a stored conversation (and optionally a second one to compare with).

Order on the page:

1. **Stat tiles**: context, KV cache, GPU memory, decode speed, TTFT and overhead, each with a sparkline.
2. **What changed** from the version before (lineage data, from the two configs).
3. **Each feature at work**: one section per analyzer that ran, measured from this conversation.
4. **KV, context and memory per turn**: predicted vs measured, with the per-layer heatmap.
5. **Compared with the version before**: the same conversation set, side by side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from innards.features import get as get_analyzer
from innards.features.base import fmt_bytes, line, section
from innards.store import RunStore

__all__ = ["analysis", "stat_tiles", "what_changed_for"]

_FEATURE_ORDER = ("gated_deltanet", "gated_attention", "moe", "mhc", "csa_hca", "dsa", "mla")


def _get(row: Any, *path: str) -> Any:
    value = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _series(turns: Sequence[Mapping[str, Any]], *path: str) -> list[Any]:
    return [_get(t, *path) for t in turns]


def stat_tiles(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Six tiles: the last turn's value plus a sparkline over turns."""
    mem = [(_get(t, "memory", "peak_bytes") or _get(t, "memory", "allocated_bytes")) for t in turns]
    specs = [
        ("context", "Context", _series(turns, "tokens", "context_after"), "tokens"),
        ("kv", "KV cache", _series(turns, "kv", "measured_bytes"), "bytes"),
        ("memory", "GPU memory" if _get(turns[-1], "memory", "source") == "torch_cuda" else "Memory", mem, "bytes"),
        ("decode", "Decode", _series(turns, "timing", "decode_tokens_per_s"), "tok/s"),
        ("ttft", "TTFT", _series(turns, "timing", "ttft_ms"), "ms"),
        ("overhead", "Overhead", _series(turns, "observer", "overhead_pct"), "%"),
    ]
    tiles = []
    for key, label, values, unit in specs:
        last = next((v for v in reversed(values) if v is not None), None)
        tiles.append(
            {
                "id": key,
                "label": label,
                "value": last,
                "unit": unit,
                "spark": values,
                "budget": _get(turns[-1], "memory", "budget_bytes") if key == "memory" else None,
            }
        )
    return tiles


def what_changed_for(model: Mapping[str, Any], other: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """What the newer model changed from the version before, from the lineup or from two stored configs."""
    from innards.lineage.lineup import pair_of, what_changed

    model_id = str(model.get("id") or "")
    found = pair_of(model_id)
    try:
        if found is not None:
            _pair, role, partner = found
            newer, previous = (model_id, partner) if role == "newer" else (partner, model_id)
            result = what_changed(newer, previous)
            result["newer_id"], result["previous_id"] = newer, previous
            return result
        if other is not None and model.get("config") and other.get("config"):
            result = what_changed(dict(model["config"]), dict(other["config"]))
            result["newer_id"], result["previous_id"] = model_id, str(other.get("id") or "")
            return result
    except Exception:
        return None
    return None


def _compare(
    a: Sequence[Mapping[str, Any]], ma: Mapping[str, Any], b: Sequence[Mapping[str, Any]], mb: Mapping[str, Any]
) -> dict[str, Any]:
    ida, idb = str(ma.get("id")), str(mb.get("id"))
    name_a, name_b = ida.rsplit("/", 1)[-1], idb.rsplit("/", 1)[-1]
    n = min(len(a), len(b))
    x = list(range(1, max(len(a), len(b)) + 1))
    rows = []
    for i in range(max(len(a), len(b))):
        ta = a[i] if i < len(a) else {}
        tb = b[i] if i < len(b) else {}
        rows.append(
            {
                "turn": i + 1,
                "message": ta.get("message") or tb.get("message"),
                "a": {
                    "context": _get(ta, "tokens", "context_after"),
                    "kv": _get(ta, "kv", "measured_bytes"),
                    "decode": _get(ta, "timing", "decode_tokens_per_s"),
                    "ttft": _get(ta, "timing", "ttft_ms"),
                    "reply": _get(ta, "output", "text"),
                },
                "b": {
                    "context": _get(tb, "tokens", "context_after"),
                    "kv": _get(tb, "kv", "measured_bytes"),
                    "decode": _get(tb, "timing", "decode_tokens_per_s"),
                    "ttft": _get(tb, "timing", "ttft_ms"),
                    "reply": _get(tb, "output", "text"),
                },
            }
        )
    findings = []
    if n:
        ka, kb = _get(a[n - 1], "kv", "measured_bytes"), _get(b[n - 1], "kv", "measured_bytes")
        if ka and kb:
            findings.append(
                f"After turn {n}, {name_a} holds {fmt_bytes(ka)} of cache and {name_b} {fmt_bytes(kb)} "
                f"({(ka - kb) / kb * 100:+.0f}%)."
            )
        da = [v for v in _series(a[:n], "timing", "decode_tokens_per_s") if v]
        db = [v for v in _series(b[:n], "timing", "decode_tokens_per_s") if v]
        if da and db:
            findings.append(
                f"Median decode speed: {name_a} {sorted(da)[len(da) // 2]:.1f} tok/s, "
                f"{name_b} {sorted(db)[len(db) // 2]:.1f} tok/s."
            )
        ca = sum(int(_get(t, "tokens", "cached") or 0) for t in a[1:n])
        pa = sum(int(_get(t, "tokens", "prompt") or 0) for t in a[1:n])
        cb = sum(int(_get(t, "tokens", "cached") or 0) for t in b[1:n])
        pb = sum(int(_get(t, "tokens", "prompt") or 0) for t in b[1:n])
        if pa and pb:
            findings.append(
                f"Prompt tokens reused from the cache after turn 1: {name_a} {ca / pa * 100:.0f}%, "
                f"{name_b} {cb / pb * 100:.0f}%."
            )
    sets = {ma.get("set"), mb.get("set")}
    if len(sets) > 1:
        findings.append("These conversations ran different sets, so turn-by-turn numbers are not like for like.")
    charts = [
        line(
            "Decode speed per turn",
            x,
            [
                {"name": name_a, "model": ida, "values": _series(a, "timing", "decode_tokens_per_s")},
                {"name": name_b, "model": idb, "values": _series(b, "timing", "decode_tokens_per_s")},
            ],
            y_label="tokens/s",
            unit="tok/s",
        ),
        line(
            "Time to first token per turn",
            x,
            [
                {"name": name_a, "model": ida, "values": _series(a, "timing", "ttft_ms")},
                {"name": name_b, "model": idb, "values": _series(b, "timing", "ttft_ms")},
            ],
            y_label="ms",
            unit="ms",
        ),
    ]
    out = section("compare", f"Compared with {name_b}", f"{name_a} vs {name_b}", findings, charts, kind="compare")
    out["rows"] = rows
    out["models"] = [ida, idb]
    return out


def analysis(
    store: RunStore,
    run: str,
    conversation: str,
    *,
    compare: tuple[str, str] | None = None,
    first: int | None = None,
    last: int | None = None,
) -> dict[str, Any]:
    """Everything the Analysis tab shows for one conversation (``compare``: another ``(run, conversation)``)."""
    record = store.read_conversation(run, conversation)
    if record is None:
        raise KeyError(f"No conversation {conversation!r} in run {run!r}")
    model = dict(record.get("model") or {})
    model["config"] = record.get("config") or {}
    model["set"] = record.get("set")
    turns = store.turns(run, conversation)
    lo, hi = (first or 1), (last or 10**9)
    turns = [t for t in turns if lo <= int(t.get("turn") or 0) <= hi]
    other: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
    if compare is not None:
        other_record = store.read_conversation(*compare)
        if other_record is not None:
            other_model = dict(other_record.get("model") or {})
            other_model["config"] = other_record.get("config") or {}
            other_model["set"] = other_record.get("set")
            other_turns = [t for t in store.turns(*compare) if lo <= int(t.get("turn") or 0) <= hi]
            other = (other_turns, other_model)
    out: dict[str, Any] = {
        "run": run,
        "conversation": conversation,
        "model": {k: v for k, v in model.items() if k != "config"},
        "set": record.get("set"),
        "status": record.get("status"),
        "turns": len(turns),
        "tiles": stat_tiles(turns) if turns else [],
        "changed": what_changed_for(model, other[1] if other else None),
        "features": record.get("feature_status") or {},
        "sections": [],
        "compare": None,
    }
    if not turns:
        return out
    present = {name for t in turns for name in (t.get("features") or {})}
    ordered = [n for n in _FEATURE_ORDER if n in present] + sorted(present - set(_FEATURE_ORDER) - {"kv"})
    for name in ordered:
        try:
            analyzer = get_analyzer(name)
            built = analyzer.conversation(turns, model, other)
        except Exception as exc:  # one broken section must not break the tab
            built = section(name, name, "could not be analyzed", [f"{type(exc).__name__}: {exc}"])
        if built:
            out["sections"].append(built)
    try:
        kv = get_analyzer("kv").conversation(turns, model, other)
    except Exception as exc:
        kv = section("kv", "KV, context and memory per turn", "could not be analyzed", [str(exc)])
    if kv:
        out["sections"].append(kv)
    if other is not None and other[0]:
        out["compare"] = _compare(turns, model, other[0], other[1])
    return out
