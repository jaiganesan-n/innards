"""KV and context: every model. Reads the cache object after each turn (shapes and dtypes only).

Shows KV per layer and per layer type, how it grows, the context, and predicted (from the config)
against measured (from the cache), with any gap explained by the engine's layout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from innards.features.base import Analyzer, Detection, Payload, TurnData, fmt_bytes, heatmap, line, section

__all__ = ["KVAnalyzer", "explain_gap"]

_GROUP_NAMES = {
    "global": "growing attention K/V",
    "sliding": "sliding-window K/V",
    "recurrent": "fixed recurrent state",
    "compressed": "compressed entries",
    "indexer": "indexer keys",
    "pending": "tokens waiting to be compressed",
}


def _get(row: Any, *path: str) -> Any:
    value = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def explain_gap(measured: Mapping[str, int], predicted: Mapping[str, int], model_type: str | None) -> list[str]:
    """Why measured bytes differ from the architectural prediction, group by group."""
    notes = []
    for group in sorted(set(measured) | set(predicted)):
        diff = int(measured.get(group, 0)) - int(predicted.get(group, 0))
        if not diff:
            continue
        if group == "pending":
            notes.append(
                f"{fmt_bytes(diff)} are tokens the engine holds until a compression window fills; they are not "
                "part of the architectural KV."
            )
        elif group == "sliding" and "compressed" in predicted:
            notes.append(
                f"{fmt_bytes(diff)} more in the window: transformers keeps separate K and V tensors on the "
                "sliding-only layers, where the architecture shares one K=V vector."
            )
        elif group == "global" and model_type == "deepseek_v32":
            ratio = measured[group] / predicted[group] if predicted.get(group) else None
            notes.append(
                "transformers caches DeepSeek-V3.2's decompressed per-head K/V instead of the latent"
                + (f" ({ratio:.1f}× the architectural bytes)." if ratio else ".")
            )
        elif group == "recurrent":
            notes.append(
                f"The recurrent state differs by {fmt_bytes(abs(diff))}: the engine holds it in another dtype "
                "than the one predicted."
            )
        else:
            name = _GROUP_NAMES.get(group, group)
            notes.append(
                f"{name.capitalize()}: measured {fmt_bytes(measured.get(group))}, predicted "
                f"{fmt_bytes(predicted.get(group))}."
            )
    return notes


class KVAnalyzer(Analyzer):
    name = "kv"
    title = "KV and context"
    reads = "Cache object after each turn (shapes and dtypes)"
    shows = "KV per layer and layer type, growth, context, predicted vs measured"
    light = True

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        from innards.kv.config import spec_from_config

        spec = spec_from_config(config)
        return Detection(
            self.name,
            tuple(layer.index for layer in spec.layers if layer.has_kv or layer.is_recurrent),
            {
                "architecture": spec.architecture,
                "attention": spec.attention,
                "layer_types": list(spec.layer_types),
                "max_context_tokens": spec.max_context_tokens,
            },
        )

    def analyze(self, turn: TurnData) -> Payload | None:
        kv = turn.record.get("kv") or {}
        measured, predicted = kv.get("measured_bytes"), kv.get("predicted_bytes")
        groups_m = kv.get("by_layer_type") or {}
        groups_p = kv.get("predicted_by_layer_type") or {}
        tokens = kv.get("cache_tokens")
        findings = []
        if measured is not None and predicted is not None:
            if measured == predicted:
                findings.append(f"Measured {fmt_bytes(measured)} for {tokens:,} tokens, exactly as predicted.")
            else:
                gap = (measured - predicted) / predicted * 100 if predicted else None
                findings.append(
                    f"Measured {fmt_bytes(measured)}, predicted {fmt_bytes(predicted)}"
                    + (f" ({gap:+.1f}%)." if gap is not None else ".")
                )
                findings += explain_gap(groups_m, groups_p, turn.config.get("model_type"))
        growth = kv.get("growth_bytes")
        if growth:
            findings.append(f"This turn added {fmt_bytes(growth)}.")
        return self.payload(
            self.title,
            {
                "measured_bytes": measured,
                "predicted_bytes": predicted,
                "cache_tokens": tokens,
                "growth_bytes": growth,
                "match": measured == predicted if measured is not None and predicted is not None else None,
            },
            findings,
            {
                "layer_types": turn.detection.facts.get("layer_types"),
                "measured_by_group": groups_m,
                "predicted_by_group": groups_p,
            },
        )

    def conversation(
        self,
        turns: Sequence[Mapping[str, Any]],
        model: Mapping[str, Any],
        other: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if not turns:
            return None
        mid = str(model.get("id") or "model")
        runs = [(turns, mid)]
        if other is not None:
            runs.append((other[0], str(other[1].get("id") or "other")))
        x = list(range(1, max(len(t) for t, _ in runs) + 1))
        kv_series, ctx_series, mem_series = [], [], []
        for rows, name in runs:
            kv_series.append(
                {
                    "name": f"{_short(name)} measured",
                    "model": name,
                    "values": [_get(t, "kv", "measured_bytes") for t in rows],
                }
            )
            kv_series.append(
                {
                    "name": f"{_short(name)} predicted",
                    "model": name,
                    "dash": "dot",
                    "values": [_get(t, "kv", "predicted_bytes") for t in rows],
                }
            )
            ctx_series.append(
                {"name": _short(name), "model": name, "values": [_get(t, "tokens", "context_after") for t in rows]}
            )
            mem_series.append(
                {
                    "name": f"{_short(name)} peak",
                    "model": name,
                    "values": [_get(t, "memory", "peak_bytes") for t in rows],
                }
            )
        charts = [
            line("KV cache per turn", x, kv_series, y_label="bytes", unit="bytes"),
            line("Context per turn", x, ctx_series, y_label="tokens", unit="tokens"),
        ]
        per_layer = [_get(t, "kv", "per_layer_bytes") or [] for t in turns]
        if any(per_layer):
            layers = max(len(p) for p in per_layer)
            z = [[(p[i] if i < len(p) else None) for p in per_layer] for i in range(layers)]
            kinds = ((turns[-1].get("features") or {}).get("kv") or {}).get("data", {}).get("layer_types") or []
            labels = [f"{i} {_kind(kinds[i]) if i < len(kinds) else ''}".strip() for i in range(layers)]
            charts.append(
                heatmap(f"Per-layer cache: {_short(mid)} (layers × turns)", x[: len(turns)], labels, z, unit="bytes")
            )
        if any(v is not None for s in mem_series for v in s["values"]):
            charts.append(line("Peak memory per turn", x, mem_series, y_label="bytes", unit="bytes"))
        return section(self.name, "KV, context and memory per turn", *self._headline(turns), charts, kind="light")

    def _headline(self, turns: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
        last = turns[-1]
        measured = _get(last, "kv", "measured_bytes")
        predicted = _get(last, "kv", "predicted_bytes")
        tokens = _get(last, "kv", "cache_tokens")
        findings = []
        if measured is not None and tokens:
            findings.append(
                f"After turn {last.get('turn')} the cache holds {fmt_bytes(measured)} for {tokens:,} tokens."
            )
        points = [(int(_get(t, "kv", "cache_tokens") or 0), _get(t, "kv", "measured_bytes")) for t in turns]
        points = [(tok, b) for tok, b in points if tok and b is not None]
        if len(points) >= 2 and points[-1][0] > points[0][0]:
            rate = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
            findings.append(f"It grew by {fmt_bytes(rate)} per token between turn 1 and turn {last.get('turn')}.")
        fixed = int((_get(last, "kv", "by_layer_type") or {}).get("recurrent") or 0)
        if fixed and measured:
            findings.append(f"{fmt_bytes(fixed)} of it ({fixed / measured * 100:.0f}%) is fixed recurrent state.")
        matches = sum(
            1 for t in turns if _get(t, "kv", "measured_bytes") == _get(t, "kv", "predicted_bytes") is not None
        )
        if measured is not None and predicted is not None:
            if matches == len(turns):
                findings.append(f"Measured equals predicted on all {len(turns)} turns.")
            else:
                payload = (last.get("features") or {}).get("kv") or {}
                findings += [f for f in payload.get("findings", [])[1:] if "added" not in f]
        reused = [(t.get("turn"), _get(t, "tokens", "cached"), _get(t, "tokens", "prompt")) for t in turns[1:]]
        cached = sum(int(c or 0) for _, c, _ in reused)
        prompt = sum(int(p or 0) for _, _, p in reused)
        if prompt:
            findings.append(f"From turn 2 on, {cached / prompt * 100:.0f}% of prompt tokens came from the kept cache.")
        headline = f"KV {fmt_bytes(measured)}" if measured is not None else "KV not measured"
        return headline, findings


def _short(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _kind(layer_type: str) -> str:
    return {
        "full_attention": "attn",
        "sliding_attention": "slide",
        "chunked_attention": "chunk",
        "linear_attention": "state",
        "compressed_sparse_attention": "CSA",
        "heavily_compressed_attention": "HCA",
        "hybrid": "hyb",
        "none": "-",
    }.get(layer_type, layer_type[:5])
