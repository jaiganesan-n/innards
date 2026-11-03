"""Gated attention (Qwen3.5, Qwen3-Next): how open each head's sigmoid output gate is, per token.

The gated layers' ``q_proj`` emits, per head, the query and a gate of the same size. The probe takes
``sigmoid(gate)`` averaged over the head dimension: 0 means the head's output is switched off for
that token, 1 means it passes through.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from innards.features.base import (
    Analyzer,
    Detection,
    Payload,
    Recorder,
    TurnData,
    heatmap,
    line,
    rnd,
    section,
    text_config,
)
from innards.features.deltanet import layer_index, top_tokens

__all__ = ["GatedAttentionAnalyzer"]


def _attention_layers(cfg: Mapping[str, Any]) -> list[int]:
    types = cfg.get("layer_types")
    if isinstance(types, list):
        return [i for i, t in enumerate(types) if str(t) in {"full_attention", "attention", "sliding_attention"}]
    return list(range(int(cfg.get("num_hidden_layers") or 0)))


class GatedAttentionAnalyzer(Analyzer):
    name = "gated_attention"
    title = "Gated attention"
    reads = "Output gate per head and token"
    shows = "Gate openness per head and token"

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        cfg = text_config(config)
        gated = cfg.get("attn_output_gate") or cfg.get("output_gate_type")
        if not gated and str(cfg.get("model_type") or "") != "qwen3_next":  # Qwen3-Next gates without the flag
            return None
        return Detection(self.name, tuple(_attention_layers(cfg)), {"heads": cfg.get("num_attention_heads")})

    def attach(self, model: Any, detection: Detection, recorder: Recorder, layers: set[int] | None = None) -> int:
        count = 0
        for module_name, module in model.named_modules():
            layer = layer_index(module_name)
            if layer is None or (layers is not None and layer not in layers):
                continue
            if not type(module).__name__.endswith("Attention") or not hasattr(module, "q_proj"):
                continue
            config = getattr(module, "config", None)
            head_dim = int(getattr(module, "head_dim", None) or getattr(config, "head_dim", 0) or 0)
            heads = int(getattr(config, "num_attention_heads", 0) or 0)
            if not head_dim or not heads or int(module.q_proj.out_features) != 2 * heads * head_dim:
                continue  # q_proj carries no gate: this attention is not output-gated

            def gate(
                _m: Any, _i: Any, out: Any, layer: int = layer, heads: int = heads, head_dim: int = head_dim
            ) -> None:
                q = out[0]
                g = q.reshape(q.shape[0], heads, 2 * head_dim)[..., head_dim:]
                recorder.put(self.name, "open", layer, g.float().sigmoid().mean(-1))

            recorder.hook(self.name, module.q_proj, gate)
            count += 1
        return count

    def analyze(self, turn: TurnData) -> Payload | None:
        positions, series = turn.series("open")
        layers = sorted(series)
        if not layers:
            return None
        keep = self.max_series
        per_token = [series[layer].mean(-1)[-keep:].tolist() for layer in layers]
        per_head = [series[layer].mean(0).tolist() for layer in layers]
        flat = [(v, layers[i], j) for i, row in enumerate(per_head) for j, v in enumerate(row)]
        mean = statistics.fmean(v for v, _, _ in flat)
        lo, hi = min(flat), max(flat)
        closed = sum(1 for v, _, _ in flat if v < 0.1)
        words = turn.text_at(positions[-keep:])
        column = [statistics.fmean(col) for col in zip(*per_token, strict=False)]
        findings = [
            f"The output gates of {len(layers)} attention layers were {mean * 100:.0f}% open on average: each head's "
            "output is scaled before it is mixed back in, and heads switch on for the tokens that need them.",
            f"Least open: layer {lo[1]} head {lo[2]} ({lo[0] * 100:.0f}%). Most open: layer {hi[1]} head {hi[2]} "
            f"({hi[0] * 100:.0f}%). {closed} of {len(flat)} heads stayed under 10% open.",
        ]
        widest = top_tokens(words, column)
        if widest:
            findings.append(f"Gates opened widest for {', '.join(widest)}.")
        return self.payload(
            self.title,
            {"open_mean": rnd(mean), "closed_heads": closed, "heads": len(flat)},
            findings,
            {
                "layers": layers,
                "tokens": words,
                "open": [[rnd(v, 3) for v in row] for row in per_token],
                "open_heads": [[rnd(v) for v in row] for row in per_head],
            },
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
        last_turn, last = rows[-1]
        data = last["data"]
        heads = data["open_heads"]
        charts = [
            heatmap(
                f"Gate openness per head, turn {last_turn.get('turn')}",
                list(range(len(heads[0]))),
                data["layers"],
                heads,
                x_label="head",
                zmin=0,
                zmax=1,
            ),
            heatmap(
                f"Gate openness per token, turn {last_turn.get('turn')}",
                data["tokens"],
                data["layers"],
                data["open"],
                x_label="token",
                zmin=0,
                zmax=1,
            ),
            line(
                "Mean gate openness per turn",
                [int(t.get("turn") or 0) for t, _ in rows],
                [
                    {
                        "name": "open",
                        "model": model.get("id"),
                        "values": [p["summary"].get("open_mean") for _, p in rows],
                    }
                ],
                y_label="share open",
                unit="ratio",
            ),
        ]
        mean = last["summary"].get("open_mean") or 0
        return section(self.name, "Gated attention at work", f"Gates {mean * 100:.0f}% open", last["findings"], charts)
