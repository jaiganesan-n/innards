"""mHC, manifold-constrained hyper-connections (DeepSeek-V4): how the residual streams are mixed.

V4 keeps ``hc_mult`` parallel residual streams. Before each attention and MLP sublayer, a hyper-connection
computes three mappings from the streams (paper eq. 8):

* ``H_pre`` (``pre``): weights that collapse the streams into the sublayer's input;
* ``H_post`` (``post``): where the sublayer's output is written back, per stream (range 0 to 2);
* ``H_res`` (``comb``): an ``hc_mult × hc_mult`` matrix that mixes the streams, projected onto the
  doubly-stochastic manifold by Sinkhorn-Knopp, so its rows and columns each sum to 1.

The probe reads ``post`` and ``comb`` from the module's output and recomputes ``pre`` from its input
(one small matmul). Per step it keeps token means, the largest row/column-sum deviation from 1, and
the norm of each stream. The analysis reports stream mixing, the doubly-stochastic check and the
signal gain across depth, both measured (stream norms) and composed (the product of the ``H_res``).
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
from innards.features.deltanet import layer_index

__all__ = ["MHCAnalyzer"]

_SITES = ("attn", "ffn")


class MHCAnalyzer(Analyzer):
    name = "mhc"
    title = "mHC (manifold-constrained hyper-connections)"
    reads = "H_pre, H_post and H_res per layer"
    shows = "Stream mixing, doubly-stochastic check (row/column sums), signal gain across depth"

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        cfg = text_config(config)
        streams = cfg.get("hc_mult")
        if not isinstance(streams, int) or streams < 2:
            return None
        n = int(cfg.get("num_hidden_layers") or 0)
        return Detection(
            self.name,
            tuple(range(n)),
            {"streams": streams, "sinkhorn_iters": cfg.get("hc_sinkhorn_iters"), "eps": cfg.get("hc_eps")},
        )

    def attach(self, model: Any, detection: Detection, recorder: Recorder, layers: set[int] | None = None) -> int:
        import torch
        import torch.nn.functional as F

        count = 0
        for module_name, module in model.named_modules():
            if not type(module).__name__.endswith("HyperConnection"):
                continue
            layer = layer_index(module_name)
            if layer is None or (layers is not None and layer not in layers):
                continue
            leaf = module_name.rsplit(".", 1)[-1]
            site = 0 if leaf.startswith("attn") else 1
            key = layer * 2 + site

            def probe(mod: Any, inputs: Any, out: Any, key: int = key) -> None:
                streams = inputs[0][0]  # [S, H, D]
                hc = int(mod.hc_mult)
                post, comb = out[0][0].float(), out[1][0].float()
                flat = mod.input_norm(streams.flatten(1).float())
                pre = torch.sigmoid(F.linear(flat, mod.fn[:hc].float()) * mod.scale[0] + mod.base[:hc]) + mod.hc_eps
                recorder.put(self.name, "pre", key, pre.mean(0, keepdim=True))
                recorder.put(self.name, "post", key, post.mean(0, keepdim=True))
                recorder.put(self.name, "comb", key, comb.mean(0).reshape(1, -1))
                deviation = torch.stack([(comb.sum(-1) - 1).abs().amax(), (comb.sum(-2) - 1).abs().amax()]).reshape(
                    1, 2
                )
                recorder.put(self.name, "deviation", key, deviation)
                recorder.put(self.name, "norm", key, streams.float().norm(dim=-1).mean(0, keepdim=True))

            recorder.hook(self.name, module, probe)
            count += 1
        return count

    def _weighted(self, turn: TurnData, name: str) -> dict[int, Any]:
        """Token-weighted mean over the turn's recorded steps (the prefill step stands for its tokens)."""
        out = {}
        for key, items in (turn.captures.get(name) or {}).items():
            total, weight = None, 0
            for step, tensor in items:
                w = max(turn.prefill_tokens, 1) if step == 0 else 1
                value = tensor.reshape(-1).double() * w
                total = value if total is None else total + value
                weight += w
            if total is not None and weight:
                out[key] = total / weight
        return out

    def analyze(self, turn: TurnData) -> Payload | None:
        import torch

        comb = self._weighted(turn, "comb")
        if not comb:
            return None
        streams = int(turn.detection.facts.get("streams") or round(len(next(iter(comb.values()))) ** 0.5))
        pre = self._weighted(turn, "pre")
        post = self._weighted(turn, "post")
        norms = self._weighted(turn, "norm")
        deviation = {
            k: max(float(t.reshape(-1).abs().max()) for _, t in items)
            for k, items in (turn.captures.get("deviation") or {}).items()
        }
        keys = sorted(comb)
        layers = sorted({k // 2 for k in keys})
        mixing = {k: 1 - float(torch.trace(comb[k].reshape(streams, streams))) / streams for k in keys}
        worst = max(deviation.values()) if deviation else None
        # composed residual map across depth: hidden <- comb^T @ hidden at every site
        composite = torch.eye(streams, dtype=torch.float64)
        gains = []
        for k in keys:
            composite = comb[k].reshape(streams, streams).T @ composite
            gains.append(float(composite.abs().sum(-1).max()))
        stream_norm = [float(norms[2 * layer].mean()) for layer in layers if 2 * layer in norms]
        measured_gain = stream_norm[-1] / stream_norm[0] if len(stream_norm) >= 2 and stream_norm[0] else None
        findings = []
        if worst is not None:
            findings.append(
                f"H_res is doubly stochastic: across {len(keys)} hyper-connections, every row and column sums to 1 "
                f"within {worst:.1e}, so mixing the {streams} streams cannot amplify the residual signal."
            )
        mix = statistics.fmean(mixing.values())
        top = max(mixing, key=mixing.__getitem__)
        findings.append(
            f"On average {mix * 100:.0f}% of each stream's residual comes from the other streams; layer {top // 2} "
            f"({_SITES[top % 2]}) mixes most ({mixing[top] * 100:.0f}%)."
        )
        if gains:
            findings.append(
                f"Composed over all {len(keys)} sites, the residual map's largest row sum is {gains[-1]:.3f} "
                "(1.000 means no gain or loss across depth)."
            )
        if measured_gain is not None:
            findings.append(
                f"Measured stream norms go from {stream_norm[0]:.2f} at layer {layers[0]} to {stream_norm[-1]:.2f} at "
                f"layer {layers[-1]} (×{measured_gain:.2f}), the growth the sublayers' writes add."
            )
        pre_mean = [rnd(float(pre[k].mean())) for k in keys if k in pre]
        post_mean = [rnd(float(post[k].mean())) for k in keys if k in post]
        return self.payload(
            "mHC",
            {
                "streams": streams,
                "mixing_mean": rnd(mix),
                "max_sum_deviation": rnd(worst, 8) if worst is not None else None,
                "composite_gain": rnd(gains[-1]) if gains else None,
                "measured_gain": rnd(measured_gain),
            },
            findings,
            {
                "sites": [f"{k // 2} {_SITES[k % 2]}" for k in keys],
                "layers": layers,
                "mixing": [rnd(mixing[k]) for k in keys],
                "deviation": [rnd(deviation.get(k), 8) for k in keys],
                "composite_gain": [rnd(g) for g in gains],
                "stream_norm": [rnd(v) for v in stream_norm],
                "pre": pre_mean,
                "post": post_mean,
                "comb_last": [[rnd(v) for v in row] for row in comb[keys[-1]].reshape(streams, streams).tolist()],
                "comb": [[rnd(v) for v in comb[k].tolist()] for k in keys],
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
        mid = model.get("id")
        streams = int(last["summary"].get("streams") or 4)
        entries = [f"{i}→{j}" for i in range(streams) for j in range(streams)]
        charts = [
            heatmap(
                f"H_res per hyper-connection, turn {last_turn.get('turn')}",
                entries,
                data["sites"],
                data["comb"],
                x_label="stream from→to",
                y_label="layer site",
                zmin=0,
                zmax=1,
            ),
            line(
                "Signal gain across depth",
                data["sites"],
                [
                    {"name": "composed H_res (max row sum)", "model": mid, "values": data["composite_gain"]},
                    {"name": "stream mixing", "model": mid, "values": data["mixing"], "dash": "dot"},
                ],
                x_label="layer site",
                y_label="gain / share",
                unit="ratio",
                mode="lines",
            ),
            line(
                "Row/column-sum deviation from 1 (doubly-stochastic check)",
                data["sites"],
                [{"name": "max |sum − 1|", "model": mid, "values": data["deviation"]}],
                x_label="layer site",
                y_label="deviation",
                mode="lines",
            ),
        ]
        if data.get("stream_norm"):
            charts.append(
                line(
                    "Residual stream norm by layer",
                    data["layers"][: len(data["stream_norm"])],
                    [{"name": "mean stream norm", "model": mid, "values": data["stream_norm"]}],
                    x_label="layer",
                    y_label="norm",
                    mode="lines",
                )
            )
        mix = last["summary"].get("mixing_mean") or 0
        return section(self.name, "mHC at work", f"{mix * 100:.0f}% cross-stream mixing", last["findings"], charts)
