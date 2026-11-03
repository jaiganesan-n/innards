"""Gated DeltaNet (Qwen3.5, Qwen3-Next): write strength, retention and the fixed-size memory.

For every linear-attention layer and token the probes read the two values the layer computes itself:

* the write strength ``β = sigmoid(in_proj_b(x))``: how strongly the token is written into the memory;
* the decay ``g = −exp(A_log) · softplus(in_proj_a(x) + dt_bias)``: ``exp(g)`` is how much of the old
  memory survives the token (retention). A head's half-life is ``ln 0.5 / mean(g)`` tokens.

At turn end the recurrent states are copied to the CPU (non-blocking), so the next turn can report how
much each layer's memory was rewritten: ``‖S_t − S_{t−1}‖ / ‖S_{t−1}‖``.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from innards.features.base import (
    Analyzer,
    Detection,
    Payload,
    Recorder,
    TurnData,
    fmt_bytes,
    heatmap,
    line,
    rnd,
    section,
    text_config,
)

__all__ = ["GatedDeltaNetAnalyzer", "layer_index", "top_tokens"]

_LAYER = re.compile(r"layers\.(\d+)\.")


def layer_index(name: str) -> int | None:
    match = _LAYER.search(name)
    return int(match.group(1)) if match else None


def top_tokens(words: Sequence[str], values: Sequence[float], top: int = 5, highest: bool = True) -> list[str]:
    """The tokens with the highest (or lowest) value, quoted, skipping specials, blanks and punctuation."""
    scored = []
    for i, (word, value) in enumerate(zip(words, values, strict=False)):
        w = str(word).strip()
        after_special = i > 0 and str(words[i - 1]).strip().startswith(("<|", "<｜"))
        if w and not w.startswith(("<|", "<｜")) and not after_special and any(c.isalnum() for c in w):
            scored.append((value, w))
    scored.sort(key=lambda s: s[0], reverse=highest)
    seen: list[str] = []
    for _, word in scored:
        if word not in seen:
            seen.append(word)
        if len(seen) == top:
            break
    return [f"“{w}”" for w in seen]


def _linear_layers(cfg: Mapping[str, Any]) -> list[int]:
    types = cfg.get("layer_types")
    if isinstance(types, list):
        return [i for i, t in enumerate(types) if "linear" in str(t)]
    n = int(cfg.get("num_hidden_layers") or 0)
    interval = int(cfg.get("full_attention_interval") or 4)
    return [i for i in range(n) if (i + 1) % interval]


def recurrent_states(cache: Any) -> dict[int, Any]:
    """``layer -> recurrent state tensor`` for layers of a transformers cache that keep one."""
    import torch

    out: dict[int, Any] = {}
    for i, layer in enumerate(getattr(cache, "layers", None) or []):
        for name, value in vars(layer).items():
            if name.startswith("is_") or ("recurrent" not in name and "ssm" not in name):
                continue
            tensors = value.values() if isinstance(value, dict) else [value]
            found = [t for t in tensors if torch.is_tensor(t) and t.numel() > 1]
            if found:
                out[i] = found[0]
                break
    return out


class GatedDeltaNetAnalyzer(Analyzer):
    name = "gated_deltanet"
    title = "Gated DeltaNet"
    reads = "β and decay per token and layer; recurrent state at turn end"
    shows = "Write strength, retention, half-life, fixed state next to growing KV"

    def __init__(self, track_state: bool = True) -> None:
        self.track_state = track_state

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        cfg = text_config(config)
        if not cfg.get("linear_num_key_heads"):
            return None
        layers = _linear_layers(cfg)
        return Detection(
            self.name,
            tuple(layers),
            {
                "value_heads": cfg.get("linear_num_value_heads"),
                "key_heads": cfg.get("linear_num_key_heads"),
                "key_dim": cfg.get("linear_key_head_dim"),
                "value_dim": cfg.get("linear_value_head_dim"),
                "conv_kernel": cfg.get("linear_conv_kernel_dim"),
                "layers_total": cfg.get("num_hidden_layers"),
            },
        )

    def attach(self, model: Any, detection: Detection, recorder: Recorder, layers: set[int] | None = None) -> int:
        import torch.nn.functional as F

        count = 0
        for module_name, module in model.named_modules():
            layer = layer_index(module_name)
            if layer is None or (layers is not None and layer not in layers):
                continue
            if not type(module).__name__.endswith("GatedDeltaNet"):
                continue
            if not (hasattr(module, "in_proj_a") and hasattr(module, "in_proj_b")):
                continue

            def beta(_m: Any, _i: Any, out: Any, layer: int = layer) -> None:
                recorder.put(self.name, "beta", layer, out[0].float().sigmoid())

            def decay(_m: Any, _i: Any, out: Any, layer: int = layer, block: Any = module) -> None:
                g = -block.A_log.float().exp() * F.softplus(out[0].float() + block.dt_bias.float())
                recorder.put(self.name, "g", layer, g)

            recorder.hook(self.name, module.in_proj_b, beta)
            recorder.hook(self.name, module.in_proj_a, decay)
            count += 2
        if count and self.track_state:

            def states(cache: Any) -> dict[int, Any]:
                wanted = set(detection.layers)
                return {i: recorder.to_cpu(t) for i, t in recurrent_states(cache).items() if i in wanted}

            recorder.at_turn_end(self.name, states)
        return count

    # ------------------------------------------------------------------ worker side

    def analyze(self, turn: TurnData) -> Payload | None:
        positions, beta = turn.series("beta")
        _, g = turn.series("g")
        layers = sorted(beta)
        findings: list[str] = []
        data: dict[str, Any] = {"layers": layers}
        summary: dict[str, Any] = {}
        if layers:
            keep = self.max_series
            beta_rows = [beta[layer].mean(-1)[-keep:].tolist() for layer in layers]
            retention_rows = [g[layer].mean(-1)[-keep:].exp().tolist() for layer in layers if layer in g]
            beta_heads = [beta[layer].mean(0).tolist() for layer in layers]
            g_heads = [g[layer].mean(0).tolist() for layer in layers if layer in g]
            half_life = [[rnd(math.log(0.5) / v, 1) if v < 0 else None for v in row] for row in g_heads]
            words = turn.text_at(positions[-keep:])
            layer_means = [sum(r) / len(r) for r in beta_rows if r]
            column = [statistics.fmean(col) for col in zip(*beta_rows, strict=False)] if beta_rows else []
            data.update(
                tokens=words,
                beta=[[rnd(v, 3) for v in row] for row in beta_rows],
                retention=[[rnd(v, 4) for v in row] for row in retention_rows],
                beta_heads=[[rnd(v) for v in row] for row in beta_heads],
                retention_heads=[[rnd(math.exp(v), 5) for v in row] for row in g_heads],
                half_life=half_life,
                token_count=len(positions),
            )
            mean_beta = statistics.fmean(layer_means) if layer_means else None
            lives = [v for row in half_life for v in row if v is not None]
            summary.update(
                beta_mean=rnd(mean_beta),
                half_life_median=rnd(statistics.median(lives), 1) if lives else None,
                heads=len(beta_heads[0]) if beta_heads else 0,
            )
            if layer_means:
                hi = max(range(len(layer_means)), key=layer_means.__getitem__)
                lo = min(range(len(layer_means)), key=layer_means.__getitem__)
                findings.append(
                    f"Write strength β averaged {mean_beta:.2f} over {len(positions):,} tokens: layer {layers[hi]} "
                    f"wrote hardest ({layer_means[hi]:.2f}), layer {layers[lo]} most gently ({layer_means[lo]:.2f})."
                )
            hardest = top_tokens(words, column)
            if hardest:
                findings.append(f"Tokens written in hardest: {', '.join(hardest)}.")
            if lives:
                short = sum(1 for v in lives if v < 2)
                long = sum(1 for v in lives if v > 1000)
                median = statistics.median(lives)
                span = f"{median:,.0f} tokens" if median >= 1.5 else ("1 token" if median >= 0.5 else "under 1 token")
                findings.append(
                    f"The median head keeps half of what it wrote for {span}; "
                    f"{short} of {len(lives)} heads forget within 2 tokens and {long} keep half for over 1,000."
                )
        states = turn.extras.get(self.name) or {}
        if states:
            previous: dict[int, Any] = turn.state.get("states") or {}
            rows = []
            for layer, tensor in sorted(states.items()):
                now = tensor.float()
                before = previous.get(layer)
                change = None
                if before is not None and before.shape == now.shape:
                    base = float(before.norm())
                    change = float((now - before).norm()) / base if base else None
                rows.append(
                    {
                        "layer": layer,
                        "bytes": int(tensor.numel() * tensor.element_size()),
                        "norm": rnd(float(now.norm())),
                        "change": rnd(change),
                    }
                )
            turn.state["states"] = {layer: t.float().clone() for layer, t in states.items()}
            size = sum(r["bytes"] for r in rows)
            data["state"] = rows
            summary["state_bytes"] = size
            changes = [r["change"] for r in rows if r["change"] is not None]
            summary["state_change_mean"] = rnd(statistics.fmean(changes)) if changes else None
            findings.append(
                f"{len(rows)} layers each keep a fixed memory of {fmt_bytes(size / len(rows))} ({fmt_bytes(size)} "
                "in all), whatever the length of the conversation."
            )
            if changes:
                findings.append(
                    f"This turn rewrote the memory by {statistics.fmean(changes) * 100:.0f}% on average: the delta "
                    "rule overwrites old associations instead of appending tokens."
                )
        if not layers and not states:
            return None
        return self.payload(self.title, summary, findings, data)

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
        charts = []
        x = [int(t.get("turn") or 0) for t, _ in rows]
        layers = data.get("layers") or []
        per_turn = []
        for layer_pos in range(len(layers)):
            per_turn.append(
                [
                    rnd(statistics.fmean(p["data"]["beta"][layer_pos]))
                    if p["data"].get("beta") and layer_pos < len(p["data"]["beta"]) and p["data"]["beta"][layer_pos]
                    else None
                    for _, p in rows
                ]
            )
        if layers:
            charts.append(heatmap("Write strength β per layer (layers × turns)", x, layers, per_turn, zmin=0, zmax=1))
        if data.get("beta"):
            charts.append(
                heatmap(
                    f"β per token, turn {last_turn.get('turn')}",
                    data.get("tokens") or [],
                    layers,
                    data["beta"],
                    x_label="token",
                    zmin=0,
                    zmax=1,
                )
            )
        if data.get("half_life"):
            logged = [[rnd(math.log10(v), 2) if v else None for v in row] for row in data["half_life"]]
            charts.append(
                heatmap(
                    "Half-life per head (log10 tokens)",
                    list(range(len(logged[0]))),
                    layers,
                    logged,
                    x_label="head",
                    unit="log10 tokens",
                )
            )
        state_x = [int(t.get("turn") or 0) for t, p in rows if p["data"].get("state")]
        if state_x:
            kv_growing = [
                (t.get("kv") or {}).get("by_layer_type", {}).get("global") for t, p in rows if p["data"].get("state")
            ]
            fixed = [p["summary"].get("state_bytes") for _, p in rows if p["data"].get("state")]
            charts.append(
                line(
                    "Fixed DeltaNet memory next to growing attention K/V",
                    state_x,
                    [
                        {"name": "DeltaNet state (fixed)", "values": fixed, "model": model.get("id")},
                        {"name": "attention K/V", "values": kv_growing, "model": model.get("id"), "dash": "dot"},
                    ],
                    y_label="bytes",
                    unit="bytes",
                )
            )
            change = [p["summary"].get("state_change_mean") for _, p in rows if p["data"].get("state")]
            if any(c is not None for c in change):
                charts.append(
                    line(
                        "Memory rewritten per turn",
                        state_x,
                        [{"name": "mean change", "values": change, "model": model.get("id")}],
                        y_label="‖ΔS‖ / ‖S‖",
                        unit="ratio",
                    )
                )
        beta = last["summary"].get("beta_mean")
        headline = f"β {beta:.2f}" if beta is not None else "Fixed memory"
        return section(self.name, "Gated DeltaNet at work", headline, last["findings"], charts)
