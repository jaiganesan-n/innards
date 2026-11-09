"""The MoE layers tab, built from an expert trace folder (``<run>/experts/<name>/``).

Order on the page:

1. **Overview**: tokens per topic, where each layer's routing was read, what is measured and what derived.
2. **Experts per layer** for one topic: layer x expert, by selection, gate share or lift.
3. **Topics compared** in one layer: topic x expert.
4. **Experts a topic needs** per layer for a coverage, and the **overlap** between topics' expert sets.
5. **Specialists**: experts chosen far more for this topic than for the others.

Per token (needs torch to read the shards): one sequence's tokens x experts in a layer (or the tokens
of one conversation turn), and the tokens one expert takes most.

The Conversation tab reads the turns stored next to the trace (``conversations.jsonl``).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from innards.experts.analysis import ExpertMap, TokenRouting, _descending, load, traces
from innards.features.base import bars, heatmap, line, section

__all__ = ["conversations", "expert_tokens", "experts_page", "sequence_section", "trace_list"]

_SOURCES = {
    "experts_input": "the experts module's inputs",
    "router_output": "the router's output",
    "topk": "the block's own top-k",
}
_MEASURE = {
    "selection": "share of the topic's tokens whose top-k included the expert",
    "gate": "the expert's average share of a token's routing weight",
    "lift": "selection for this topic ÷ selection for the other topics",
}


def trace_list(root: Path, only: str | None = None) -> list[dict[str, Any]]:
    return [t for t in traces(root) if only is None or t["run"] == only]


def _pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if 0 < v < 0.1 else f"{v * 100:.0f}%"


def _overview(experts: ExpertMap) -> dict[str, Any]:
    tokens = ", ".join(f"{t}: {experts.tokens(t):,}" for t in experts.topics) or "none yet"
    sources: dict[str, int] = {}
    for info in experts.layer_info:
        key = _SOURCES.get(str(info.get("source")), "not read yet")
        sources[key] = sources.get(key, 0) + 1
    read = "; ".join(f"{name} ({n} layers)" for name, n in sources.items())
    missed = sum(int(info.get("missed") or 0) for info in experts.layer_info)
    findings = [
        f"Tokens measured per topic (padding never counted): {tokens}.",
        f"Routing was read from {read}." + (f" {missed} block forwards had no routing to read." if missed else ""),
        "Measured: how often each expert was in a token's top-k, and its share of the routing weight. "
        "Derived from those counts: selection, gate share, lift and the expert sets.",
    ]
    if any(info.get("weights") == "topk_values" for info in experts.layer_info):
        findings.append(
            "Some layers expose only the scores top-k selected on (which can include a selection bias), so "
            "their gate shares are approximate; selection is exact everywhere."
        )
    facts = experts.routing
    if facts.get("experts"):
        shared = f", {facts['shared_experts']} shared (always on)" if facts.get("shared_experts") else ""
        findings.append(f"Config: {facts['experts']} routed experts{shared}, each token goes to {facts.get('top_k')}.")
    title = f"Expert routing: {experts.model}"
    return section("experts-overview", title, f"{len(experts.layers)} MoE layers", findings)


def experts_page(
    folder: Path,
    topic: str | None = None,
    by: str = "selection",
    layer: int | None = None,
    coverage: float = 0.9,
) -> dict[str, Any]:
    experts = load(folder)
    sections = [_overview(experts)]
    topic = topic if topic in experts.topics else (experts.topics[0] if experts.topics else None)
    layer = layer if layer in experts.layers else (experts.layers[len(experts.layers) // 2] if experts.layers else None)
    out: dict[str, Any] = {
        "model": experts.model,
        "topics": experts.topics,
        "layers": experts.layers,
        "topic": topic,
        "layer": layer,
        "by": by,
        "coverage": coverage,
        "tokens": {t: experts.tokens(t) for t in experts.topics},
        "has_tokens": bool(experts.data.get("shards")),
        "sections": sections,
    }
    if topic is None or layer is None:
        return out
    width = max(experts.experts(m) for m in experts.layers)
    values = experts.metric(topic, by)
    z = [row + [None] * (width - len(row)) for row in values]
    needed = experts.needed(topic, coverage, by="gate" if by == "gate" else "selection")
    per_layer = []
    for li, layer_id in enumerate(experts.layers):
        row = values[li]
        best = min(range(len(row)), key=lambda e: _descending(row[e]))
        per_layer.append((row[best] or 0, layer_id, best))
    busiest = sorted(per_layer, reverse=True)[:3]
    findings = [
        f"{by.capitalize()}: {_MEASURE[by]} (derived).",
        f"Covering {coverage:.0%} of {topic}'s routing takes {statistics.fmean(needed):.1f} experts per layer on "
        f"average (from {min(needed)} to {max(needed)}) of {width}.",
        "Highest: "
        + "; ".join(
            f"layer {la} expert {e} ({v:.2f}×)" if by == "lift" else f"layer {la} expert {e} ({_pct(v)})"
            for v, la, e in busiest
        )
        + ".",
        "Click a cell to see the tokens that expert takes.",
    ]
    sections.append(
        section(
            "experts-layers",
            f"Experts per layer: {topic}",
            f"{experts.tokens(topic):,} tokens",
            findings,
            [
                heatmap(
                    f"{by} by layer and expert ({topic})",
                    list(range(width)),
                    experts.layers,
                    z,
                    x_label="expert",
                    y_label="layer",
                    unit="ratio",
                    zmin=0,
                    zmax=1 if by != "lift" else None,
                )
                | {"click": "expert"}
            ],
        )
    )
    if len(experts.topics) > 1:
        li = experts.layers.index(layer)
        rows = [experts.metric(t, by)[li] for t in experts.topics]
        overlap = [
            [
                statistics.fmean([v for v in experts.overlap(a, b, coverage) if v is not None] or [0.0])
                for b in experts.topics
            ]
            for a in experts.topics
        ]
        sections.append(
            section(
                "experts-topics",
                f"Topics compared: layer {layer}",
                f"{len(experts.topics)} topics",
                [
                    f"Each row is one topic's {by} over the experts of layer {layer} (derived).",
                    f"Overlap: Jaccard of two topics' expert sets at {coverage:.0%} coverage, averaged over layers; "
                    "1 means the same experts, 0 none in common.",
                ],
                [
                    heatmap(
                        f"{by} by topic, layer {layer}",
                        list(range(len(rows[0]))),
                        experts.topics,
                        rows,
                        x_label="expert",
                        y_label="topic",
                        unit="ratio",
                        zmin=0,
                    )
                    | {"click": "expert", "layer": layer},
                    heatmap(
                        f"Overlap of expert sets ({coverage:.0%} coverage)",
                        experts.topics,
                        experts.topics,
                        overlap,
                        x_label="topic",
                        y_label="topic",
                        unit="ratio",
                        zmin=0,
                        zmax=1,
                    ),
                ],
            )
        )
    measure = "gate" if by == "gate" else "selection"
    series = [
        {"name": t, "values": experts.needed(t, coverage, by=measure), "dash": None if t == topic else "dot"}
        for t in experts.topics
    ]
    chosen = experts.select(topic, coverage, by=measure)
    sections.append(
        section(
            "experts-needed",
            f"Experts {topic} needs ({coverage:.0%} coverage)",
            f"keep {chosen.kept} of {chosen.total}",
            [
                f"Per layer, the fewest experts that take {coverage:.0%} of {topic}'s routing by {measure} "
                f"(derived from {chosen.tokens:,} measured tokens). Download the set to build a smaller model.",
                "Shared experts, when the model has them, run for every token and are always kept.",
            ],
            [
                line(
                    f"Experts needed per layer for {coverage:.0%} coverage",
                    experts.layers,
                    series,
                    x_label="layer",
                    y_label="experts",
                    unit="count",
                ),
                bars(
                    f"Experts kept per layer: {topic}",
                    experts.layers,
                    [
                        {"name": "kept", "values": [len(r["keep"]) for r in chosen.layers]},
                        {"name": "not needed", "values": [r["of"] - len(r["keep"]) for r in chosen.layers]},
                    ],
                    x_label="layer",
                    y_label="experts",
                    unit="count",
                    stacked=True,
                ),
            ],
        )
    )
    specialists = experts.specialists(topic, n=10)
    if specialists:
        lines = []
        for r in specialists:
            if r["only"]:
                tail = "and no token of the other topics"
            else:
                tail = f"vs {_pct(r['selection'] / r['lift'])} of the others (×{r['lift']:.1f})"
            lines.append(f"Layer {r['layer']} expert {r['expert']}: {_pct(r['selection'])} of {topic} tokens {tail}.")
        sections.append(section("experts-specialists", f"Specialists: {topic}", f"{len(specialists)} experts", lines))
    out["selection"] = chosen.to_dict()["summary"]
    return out


def conversations(folder: Path) -> list[dict[str, Any]]:
    """The Conversation tab's stored turns, grouped by conversation in the order they were held."""
    from innards.view.chat import CONVERSATIONS

    path = folder / CONVERSATIONS
    grouped: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            turn = json.loads(raw)
        except json.JSONDecodeError:
            continue  # a line cut short by a crash; the others still read
        if not isinstance(turn, dict) or not turn.get("conversation"):
            continue
        conv = grouped.setdefault(str(turn["conversation"]), {"id": str(turn["conversation"]), "turns": []})
        conv["turns"].append(turn)
    return list(grouped.values())


def sequence_section(
    folder: Path, sequence: int, layer: int, first: int | None = None, last: int | None = None
) -> dict[str, Any]:
    """One sequence's tokens x experts in ``layer``; ``first``/``last`` keep positions ``first <= p < last``."""
    routing = TokenRouting(folder)
    data = routing.sequence(sequence)
    if layer not in data["layers"]:
        raise KeyError(f"No MoE layer {layer} in this trace")
    li = data["layers"].index(layer)
    width = routing.experts[li]
    lo, hi = first if first is not None else 0, last if last is not None else 2**62
    keep = [i for i, p in enumerate(data["positions"]) if lo <= p < hi]
    if not keep:
        raise KeyError(f"Sequence {sequence} has no tokens at positions {lo} to {hi - 1}")
    for key in ("ids", "text", "positions", "experts", "weights", "topics"):
        data[key] = [data[key][i] for i in keep]
    z: list[list[float | None]] = [[None] * len(data["ids"]) for _ in range(width)]
    for t, (experts, weights) in enumerate(zip(data["experts"], data["weights"], strict=True)):
        chosen = [(e, w) for e, w in zip(experts[li], weights[li], strict=True) if e >= 0]
        total = sum(abs(w) for _, w in chosen if w is not None) or 1.0
        for e, w in chosen:
            z[e][t] = abs(w) / total if w is not None else 1.0 / len(chosen)
    words = [t.replace("\n", "↵") or "·" for t in data["text"]]
    sliced = first is not None or last is not None
    span = f", positions {data['positions'][0]} to {data['positions'][-1]}" if sliced else ""
    topics = " + ".join(dict.fromkeys(str(t) for t in data["topics"]))
    return section(
        "experts-sequence",
        f"Tokens and their experts: layer {layer}",
        f"{topics} · {len(words)} tokens{span}",
        [
            f"Each column is one token; colored cells are the experts its top-k chose in layer {layer}, shaded by "
            "the share of the token's routing weight (measured).",
        ],
        [
            heatmap(
                f"Sequence {sequence}{span}, layer {layer}",
                words,
                list(range(width)),
                z,
                x_label="token",
                y_label="expert",
                unit="ratio",
                zmin=0,
                zmax=1,
            )
        ],
    )


def expert_tokens(folder: Path, layer: int, expert: int, topic: str | None = None) -> dict[str, Any]:
    rows = TokenRouting(folder).expert(layer, expert, n=25, topic=topic)
    scope = f" in {topic}" if topic else ""
    findings = [
        f"{row['text']!r}: {row['routed']:,} of {row['occurrences']:,} occurrences ({_pct(row['share'])})"
        for row in rows
    ] or [f"No recorded token{scope} went to this expert."]
    return section(
        "experts-expert",
        f"Layer {layer} expert {expert}: the tokens it takes{scope}",
        f"{len(rows)} tokens",
        ["Counted from every recorded token (measured): times routed here, of all occurrences.", *findings],
    )
