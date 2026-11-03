"""Cache-layout analyzers: CSA/HCA (DeepSeek-V4), DSA (DeepSeek-V3.2) and MLA (DeepSeek-V2/V3).

These read the cache object after each turn (shapes and dtypes only, like the KV analyzer), so they
attach no hooks and cost nothing during generation.

* **CSA/HCA**: how many compressed entries, indexer keys and window tokens each layer holds, against
  ``⌊T / ratio⌋``, and the tokens the engine keeps waiting for the next window.
* **DSA**: the lightning-indexer key cache next to the main KV, and the share of the context each
  query can attend to (``index_topk / T``).
* **MLA**: whether the engine caches the latent (``kv_lora_rank + qk_rope_head_dim`` per token and
  layer) or decompressed per-head K/V, and the ratio between the two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from innards.features.base import (
    Analyzer,
    Detection,
    Payload,
    TurnData,
    bars,
    fmt_bytes,
    line,
    rnd,
    section,
    text_config,
)

__all__ = ["CompressedAttentionAnalyzer", "DSAAnalyzer", "MLAAnalyzer"]


def _spec(config: Mapping[str, Any]) -> Any:
    from innards.kv.config import spec_from_config

    return spec_from_config(config)


def _elements(meta: Sequence[Any]) -> int:
    shape = meta[0] if len(meta) == 2 else meta[1]
    return math.prod(int(d) for d in shape)


def _bytes(meta: Sequence[Any]) -> int:
    shape, itemsize = (meta[0], meta[1]) if len(meta) == 2 else (meta[1], meta[2])
    return math.prod(int(d) for d in shape) * int(itemsize)


def _get(row: Any, *path: str) -> Any:
    value = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


class CompressedAttentionAnalyzer(Analyzer):
    name = "csa_hca"
    title = "Compressed attention (CSA/HCA)"
    reads = "Cache layout: window, compressed entries, indexer keys"
    shows = "Compressed vs window entries, indexer cache, tokens waiting to be compressed"
    light = True

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        spec = _spec(config)
        compressed = [layer for layer in spec.layers if layer.is_compressed]
        if not compressed:
            return None
        cfg = text_config(config)
        return Detection(
            self.name,
            tuple(layer.index for layer in spec.layers if layer.has_kv),
            {
                "window": spec.sliding_window,
                "ratios": {str(layer.index): layer.compress_ratio for layer in compressed},
                "csa_layers": [layer.index for layer in compressed if layer.indexer_elements],
                "hca_layers": [layer.index for layer in compressed if not layer.indexer_elements],
                "index_topk": cfg.get("index_topk"),
                "index_head_dim": cfg.get("index_head_dim"),
            },
        )

    def analyze(self, turn: TurnData) -> Payload | None:
        if not turn.cache_layers:
            return None
        facts = turn.detection.facts
        ratios = {int(k): int(v) for k, v in (facts.get("ratios") or {}).items()}
        tokens = int(_get(turn.record, "kv", "cache_tokens") or 0)
        rows: list[dict[str, Any]] = []
        for i, layer in enumerate(turn.cache_layers):
            extra = layer.get("extra") or []
            entries = max((int(m[1][1]) for m in extra if m[0] == "compressed" and len(m[1]) >= 3), default=0)
            indexer = max((int(m[1][1]) for m in extra if m[0] == "indexer" and len(m[1]) >= 3), default=0)
            pending = max((int(m[1][1]) for m in extra if m[0] == "pending" and len(m[1]) >= 3), default=0)
            kv = layer.get("kv") or []
            window = int(kv[0][0][-2]) if kv and len(kv[0][0]) >= 3 else 0
            ratio = ratios.get(i)
            rows.append(
                {
                    "layer": i,
                    "ratio": ratio,
                    "window_tokens": window,
                    "entries": entries,
                    "entries_predicted": tokens // ratio if ratio else 0,
                    "indexer_entries": indexer,
                    "pending_tokens": pending,
                    "bytes": {
                        "window": sum(_bytes(m) for m in kv),
                        "compressed": sum(_bytes(m) for m in extra if m[0] == "compressed"),
                        "indexer": sum(_bytes(m) for m in extra if m[0] == "indexer"),
                        "pending": sum(_bytes(m) for m in extra if m[0] == "pending"),
                    },
                }
            )
        totals = {k: sum(r["bytes"][k] for r in rows) for k in ("window", "compressed", "indexer", "pending")}
        csa = [r for r in rows if r["layer"] in set(facts.get("csa_layers") or [])]
        hca = [r for r in rows if r["layer"] in set(facts.get("hca_layers") or [])]
        exact = all(r["entries"] == r["entries_predicted"] for r in rows if r["ratio"])
        findings = [
            f"At {tokens:,} tokens every layer keeps a window of {max((r['window_tokens'] for r in rows), default=0)} "
            f"tokens ({fmt_bytes(totals['window'])} in all)."
        ]
        if csa:
            findings.append(
                f"{len(csa)} CSA layers hold {csa[0]['entries']:,} compressed entries each (one per "
                f"{csa[0]['ratio']} tokens) and as many indexer keys ({fmt_bytes(totals['indexer'])} of indexer "
                "cache in all)."
            )
        if hca:
            findings.append(
                f"{len(hca)} HCA layers hold {hca[0]['entries']:,} entries each (one per {hca[0]['ratio']} tokens)."
            )
        findings.append(
            "Compressed entries match ⌊T / ratio⌋ on every layer."
            if exact
            else "Some layers hold a different number of compressed entries than ⌊T / ratio⌋."
        )
        if totals["pending"]:
            findings.append(
                f"{fmt_bytes(totals['pending'])} are tokens waiting to fill the next window (engine buffers)."
            )
        topk = facts.get("index_topk")
        if topk and csa and csa[0]["entries"]:
            share = min(int(topk) / csa[0]["entries"], 1.0)
            findings.append(
                f"The lightning indexer lets each query attend to at most {topk} compressed entries: "
                f"{share * 100:.0f}% of the {csa[0]['entries']:,} a CSA layer holds now."
            )
        return self.payload(
            self.title,
            {"cache_tokens": tokens, "entries_match": exact, **{f"{k}_bytes": v for k, v in totals.items()}},
            findings,
            {"layers": rows},
        )

    def conversation(
        self,
        turns: Sequence[Mapping[str, Any]],
        model: Mapping[str, Any],
        other: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        rows = self.payloads(turns, self.name)
        if not rows:
            return None
        x = [int(t.get("turn") or 0) for t, _ in rows]
        parts = ("window", "compressed", "indexer", "pending")
        charts = [
            bars(
                "Cache by part per turn",
                x,
                [{"name": part, "values": [p["summary"].get(f"{part}_bytes") for _, p in rows]} for part in parts],
                x_label="turn",
                y_label="bytes",
                unit="bytes",
                stacked=True,
            ),
        ]
        last_turn, last = rows[-1]
        layers = last["data"]["layers"]
        charts.append(
            line(
                f"Compressed entries per layer, turn {last_turn.get('turn')}",
                [r["layer"] for r in layers],
                [
                    {"name": "measured", "model": model.get("id"), "values": [r["entries"] for r in layers]},
                    {
                        "name": "⌊T / ratio⌋",
                        "model": model.get("id"),
                        "dash": "dot",
                        "values": [r["entries_predicted"] for r in layers],
                    },
                ],
                x_label="layer",
                y_label="entries",
                mode="markers",
            )
        )
        headline = f"{fmt_bytes(last['summary'].get('compressed_bytes'))} compressed"
        return section(self.name, "Compressed attention at work", headline, last["findings"], charts)


class DSAAnalyzer(Analyzer):
    name = "dsa"
    title = "DeepSeek Sparse Attention (lightning indexer)"
    reads = "Cache layout: indexer keys next to the main KV"
    shows = "Indexer cache, share of the context each query attends to"
    light = True

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        spec = _spec(config)
        indexed = [layer for layer in spec.layers if layer.indexer_elements and not layer.is_compressed]
        if not indexed:
            return None
        cfg = text_config(config)
        return Detection(
            self.name,
            tuple(layer.index for layer in indexed),
            {
                "index_topk": cfg.get("index_topk"),
                "index_head_dim": cfg.get("index_head_dim"),
                "index_heads": cfg.get("index_n_heads"),
            },
        )

    def analyze(self, turn: TurnData) -> Payload | None:
        tokens = int(_get(turn.record, "kv", "cache_tokens") or 0)
        indexer = sum(_bytes(m) for layer in turn.cache_layers for m in (layer.get("extra") or []) if m[0] == "indexer")
        main = sum(_bytes(m) for layer in turn.cache_layers for m in (layer.get("kv") or []))
        topk = turn.detection.facts.get("index_topk")
        share = min(int(topk) / tokens, 1.0) if topk and tokens else None
        findings = []
        if indexer:
            findings.append(
                f"The indexer keeps {fmt_bytes(indexer)} of keys ({turn.detection.facts.get('index_head_dim')} per "
                f"token and layer) next to {fmt_bytes(main)} of attention cache."
            )
        if share is not None:
            findings.append(
                f"Each query attends to at most {topk} tokens: {share * 100:.0f}% of the {tokens:,} in context"
                + (" (all of them until the context passes top-k)." if share >= 1 else ".")
            )
        return self.payload(
            self.title,
            {"indexer_bytes": indexer, "main_bytes": main, "attended_share": rnd(share), "cache_tokens": tokens},
            findings,
            {"index_topk": topk},
        )

    def conversation(
        self,
        turns: Sequence[Mapping[str, Any]],
        model: Mapping[str, Any],
        other: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        rows = self.payloads(turns, self.name)
        if not rows:
            return None
        x = [int(t.get("turn") or 0) for t, _ in rows]
        charts = [
            line(
                "Share of the context each query attends to",
                x,
                [
                    {
                        "name": "top-k / context",
                        "model": model.get("id"),
                        "values": [p["summary"].get("attended_share") for _, p in rows],
                    }
                ],
                unit="ratio",
            ),
            bars(
                "Indexer keys next to the attention cache",
                x,
                [
                    {"name": "attention cache", "values": [p["summary"].get("main_bytes") for _, p in rows]},
                    {"name": "indexer keys", "values": [p["summary"].get("indexer_bytes") for _, p in rows]},
                ],
                x_label="turn",
                unit="bytes",
                stacked=True,
            ),
        ]
        last = rows[-1][1]
        return section(
            self.name,
            "Sparse attention at work",
            f"{fmt_bytes(last['summary'].get('indexer_bytes'))} indexer cache",
            last["findings"],
            charts,
        )


class MLAAnalyzer(Analyzer):
    name = "mla"
    title = "Multi-head latent attention"
    reads = "Cache layout: latent or expanded K/V"
    shows = "Latent vs expanded layout, elements per token and layer"
    light = True

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        cfg = text_config(config)
        rank = cfg.get("kv_lora_rank")
        if not isinstance(rank, int) or not rank:
            return None
        spec = _spec(config)
        heads = int(cfg.get("num_attention_heads") or 0)
        rope = int(cfg.get("qk_rope_head_dim") or 0)
        nope = int(cfg.get("qk_nope_head_dim") or 0)
        v_dim = int(cfg.get("v_head_dim") or 0)
        return Detection(
            self.name,
            tuple(layer.index for layer in spec.layers if layer.has_kv),
            {
                "latent": rank + rope,
                "rank": rank,
                "rope": rope,
                "heads": heads,
                "expanded": heads * (nope + rope) + heads * v_dim,
            },
        )

    def analyze(self, turn: TurnData) -> Payload | None:
        tokens = int(_get(turn.record, "kv", "cache_tokens") or 0)
        facts = turn.detection.facts
        layers = [layer for layer in turn.cache_layers if layer.get("kv")]
        if not layers or not tokens:
            return None
        per_layer = sum(_elements(m) for m in layers[0]["kv"]) / tokens
        latent, expanded = int(facts["latent"]), int(facts["expanded"])
        layout = "latent" if per_layer <= latent * 1.5 else "expanded"
        if layout == "latent":
            finding = (
                f"The engine caches the latent: {per_layer:,.0f} elements per token and layer "
                f"({facts['rank']} latent + {facts['rope']} RoPE key), {expanded / latent:.0f}× less than per-head "
                f"K/V ({expanded:,})."
            )
        else:
            finding = (
                f"The engine caches decompressed per-head K/V: {per_layer:,.0f} elements per token and layer, "
                f"{per_layer / latent:.0f}× the {latent}-element latent the architecture needs."
            )
        return self.payload(
            self.title,
            {"layout": layout, "elements_per_token_layer": rnd(per_layer, 1), "latent": latent, "expanded": expanded},
            [finding],
            {"layout": layout},
        )

    def conversation(
        self,
        turns: Sequence[Mapping[str, Any]],
        model: Mapping[str, Any],
        other: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        rows = self.payloads(turns, self.name)
        if not rows:
            return None
        last = rows[-1][1]
        s = last["summary"]
        chart = bars(
            "Elements cached per token and layer",
            ["latent (architecture)", "measured", "per-head K/V"],
            [{"name": "elements", "values": [s["latent"], s["elements_per_token_layer"], s["expanded"]]}],
            unit="count",
        )
        return section(self.name, "Latent attention cache", f"{s['layout']} layout", last["findings"], [chart])
