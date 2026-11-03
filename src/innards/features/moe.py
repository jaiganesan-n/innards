"""Mixture of experts, including LatentMoE (Nemotron 3): expert load and routing entropy per turn.

The probes find every MoE block and read the experts each token was sent to with
:mod:`innards.experts.routing` (the experts module's inputs, the router's output, or the block's own
top-k for remote code that routes inline). On the device they reduce it to per-expert counts and the
entropy of each token's top-k weights, so only a few numbers per token leave the GPU. Routing by
topic, per token, and the experts a topic needs are in :mod:`innards.experts`.

LatentMoE projects each token into a smaller latent space (``moe_latent_size``) before the experts and
back afterwards, so every expert works on a vector ``hidden / latent`` times smaller.
"""

from __future__ import annotations

import functools
import math
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

__all__ = ["MoEAnalyzer"]


def _experts(cfg: Mapping[str, Any]) -> int:
    for key in ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"):
        value = cfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            return value
    return 0


def _moe_layers(cfg: Mapping[str, Any]) -> list[int]:
    blocks = cfg.get("layers_block_type")
    if isinstance(blocks, list):
        return [i for i, b in enumerate(blocks) if str(b) == "moe"]
    pattern = cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str):
        return [i for i, c in enumerate(pattern) if c == "E"]
    n = int(cfg.get("num_hidden_layers") or 0)
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    step = int(cfg.get("decoder_sparse_step") or 1)
    mlp_only = set(cfg.get("mlp_only_layers") or [])
    return [i for i in range(first_dense, n) if (i + 1) % step == 0 and i not in mlp_only]


class MoEAnalyzer(Analyzer):
    name = "moe"
    title = "Mixture of experts"
    reads = "Router logits and expert choices; latent projection size"
    shows = "Expert load, routing entropy, latent size vs hidden size"

    def detect(self, config: Mapping[str, Any]) -> Detection | None:
        cfg = text_config(config)
        experts = _experts(cfg)
        if not experts:
            return None
        latent = cfg.get("moe_latent_size")
        mlp_types = cfg.get("mlp_layer_types")
        return Detection(
            self.name,
            tuple(_moe_layers(cfg)),
            {
                "experts": experts,
                "top_k": cfg.get("num_experts_per_tok") or cfg.get("moe_topk"),
                "shared_experts": cfg.get("n_shared_experts"),
                "hidden_size": cfg.get("hidden_size"),
                "latent_size": latent,
                "latent": bool(latent),
                "hash_layers": [i for i, t in enumerate(mlp_types) if t == "hash_moe"]
                if isinstance(mlp_types, list)
                else [],
            },
        )

    def attach(self, model: Any, detection: Detection, recorder: Recorder, layers: set[int] | None = None) -> int:
        import torch

        from innards.experts.routing import MoELayer, attach, find_moe_layers

        found = find_moe_layers(model, layers)

        def route(moe: MoELayer, index: Any, weights: Any) -> None:
            size = moe.experts
            chosen = index.reshape(-1).long()
            real = ((chosen >= 0) & (chosen < size)).to(torch.float32)  # sentinel slots are not experts
            counts = torch.bincount(chosen.clamp(0, size - 1), weights=real, minlength=size)[None, :]
            recorder.put(self.name, "counts", moe.layer, counts)
            if weights is not None:
                w = weights.float()
                p = w / w.abs().sum(-1, keepdim=True).clamp_min(1e-20)
                entropy = -(p * p.clamp_min(1e-20).log()).sum(-1)
                recorder.put(self.name, "entropy", moe.layer, entropy[:, None])

        attach(found, route, functools.partial(recorder.hook, self.name))
        return len(found)

    def analyze(self, turn: TurnData) -> Payload | None:
        rows = turn.captures.get("counts") or {}
        if not rows:
            return None
        facts = turn.detection.facts
        experts = int(facts.get("experts") or 0)
        top_k = int(facts.get("top_k") or 1)
        layers = sorted(rows)
        load, balance, used, max_share = [], [], [], []
        for layer in layers:
            total = sum(t.reshape(-1)[: experts or None].double() for _, t in rows[layer])
            counts = [int(c) for c in total.tolist()]
            slots = sum(counts)
            load.append([rnd(c / slots, 5) if slots else 0.0 for c in counts])
            probs = [c / slots for c in counts if c] if slots else []
            entropy = -sum(p * math.log(p) for p in probs)
            balance.append(rnd(entropy / math.log(len(counts))) if len(counts) > 1 and slots else None)
            used.append(sum(1 for c in counts if c))
            max_share.append(rnd(max(counts) / slots) if slots else None)
        _, ent = turn.series("entropy")
        token_entropy = [float(ent[layer].mean()) for layer in sorted(ent)] if ent else []
        ideal = min(top_k / experts, 1.0) if experts else None
        findings = [
            f"Tokens went through {len(layers)} MoE layers, each to {top_k} of {experts} experts; on average "
            f"{statistics.fmean(used):.1f} different experts were used per layer this turn.",
        ]
        valid = [(s, layer) for s, layer in zip(max_share, layers, strict=True) if s is not None]
        if valid:
            share, layer = max(valid)
            findings.append(
                f"The busiest expert took {share * 100:.0f}% of the routing slots in layer {layer}"
                + (f" (an even spread would be {ideal * 100:.1f}%)." if ideal else ".")
            )
        balances = [b for b in balance if b is not None]
        if balances:
            findings.append(
                "Load balance (routing entropy over experts, 1 = perfectly even) averaged "
                f"{statistics.fmean(balances):.2f}."
            )
        if facts.get("latent"):
            hidden, latent = facts.get("hidden_size"), facts.get("latent_size")
            if hidden and latent:
                findings.append(
                    f"LatentMoE: tokens enter the experts as {latent}-dim latents instead of {hidden}-dim hidden "
                    f"states, so each expert works on a vector {hidden / latent:.0f}× smaller."
                )
        if facts.get("hash_layers"):
            findings.append(
                f"Layers {', '.join(map(str, facts['hash_layers']))} route by a fixed token-id table (hash routing)."
            )
        return self.payload(
            "LatentMoE" if facts.get("latent") else self.title,
            {
                "balance_mean": rnd(statistics.fmean(balances)) if balances else None,
                "experts_used_mean": rnd(statistics.fmean(used), 2),
                "token_entropy_mean": rnd(statistics.fmean(token_entropy)) if token_entropy else None,
                "experts": experts,
                "top_k": top_k,
            },
            findings,
            {
                "layers": layers,
                "load": load,
                "balance": balance,
                "used": used,
                "max_share": max_share,
                "latent_size": facts.get("latent_size"),
                "hidden_size": facts.get("hidden_size"),
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
        charts = [
            heatmap(
                f"Expert load, turn {last_turn.get('turn')} (share of routing slots)",
                list(range(len(data["load"][0]) if data["load"] else 0)),
                data["layers"],
                data["load"],
                x_label="expert",
                unit="ratio",
                zmin=0,
            ),
        ]
        series = [
            {
                "name": "load balance",
                "model": model.get("id"),
                "values": [p["summary"].get("balance_mean") for _, p in rows],
            }
        ]
        if other is not None:
            other_rows = self.payloads(other[0], self.name)
            if other_rows:
                series.append(
                    {
                        "name": f"{str(other[1].get('id')).rsplit('/', 1)[-1]} load balance",
                        "model": other[1].get("id"),
                        "values": [p["summary"].get("balance_mean") for _, p in other_rows],
                    }
                )
        charts.append(
            line(
                "Routing entropy over experts per turn (1 = even)",
                [int(t.get("turn") or 0) for t, _ in rows],
                series,
                y_label="balance",
                unit="ratio",
            )
        )
        balance = last["summary"].get("balance_mean")
        headline = f"balance {balance:.2f}" if balance is not None else "routing"
        return section(self.name, f"{last['title']} at work", headline, last["findings"], charts)
