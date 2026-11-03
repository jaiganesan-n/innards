"""What a trace measured, by topic, layer and expert, and which experts a topic needs.

Measured (counted from the routing the model chose):

* ``tokens``: tokens of a topic that went through a layer (padding is never counted).
* ``counts``: how often each expert was in a token's top-k.
* ``gates``: the sum, over tokens, of each expert's share of that token's routing weight.

Derived from those counts, never measured on their own, and labeled as such wherever they appear:

* **selection**: share of a topic's tokens whose top-k included the expert (``counts / tokens``, 0–1).
* **gate**: the expert's average share of a token's routing weight (``gates / tokens``; sums to 1 per layer).
* **lift**: selection for this topic divided by selection for every other topic together.
* **select(topic, coverage)**: per layer, the fewest experts that together take ``coverage`` of the
  topic's routing (by selection or by gate). This is the expert set to keep for that topic.

:class:`ExpertMap` needs only ``summary.json`` (plain Python, no torch). :class:`TokenRouting` reads the
token shards and needs torch and safetensors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["METRICS", "ExpertMap", "Selection", "TokenRouting", "load", "traces"]

METRICS = ("selection", "gate", "lift")


def _summary_path(path: str | Path) -> list[Path]:
    """``summary.json`` files under a trace folder, a run folder or its ``experts`` folder."""
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if (p / "summary.json").is_file():
        return [p / "summary.json"]
    for base in (p / "experts", p):
        found = sorted(base.glob("*/summary.json"))
        if found:
            return found
    raise FileNotFoundError(f"No expert trace (summary.json) under {p}")


def traces(root: str | Path) -> list[dict[str, Any]]:
    """Every trace in a folder of runs: ``{"run", "name", "model", "topics", "tokens", "updated_at", "path"}``."""
    out = []
    for path in sorted(Path(root).expanduser().glob("*/experts/*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tokens = [max(row) if row else 0 for row in data.get("tokens") or []]
        out.append(
            {
                "run": path.parents[2].name,
                "name": path.parent.name,
                "model": (data.get("model") or {}).get("id"),
                "topics": data.get("topics") or [],
                "tokens": dict(zip(data.get("topics") or [], tokens, strict=False)),
                "layers": len(data.get("layers") or []),
                "updated_at": data.get("updated_at"),
                "path": str(path.parent),
            }
        )
    return sorted(out, key=lambda t: str(t.get("updated_at") or ""), reverse=True)


def load(path: str | Path) -> ExpertMap:
    """An :class:`ExpertMap` from a trace folder, or a run folder (its traces of one model are added up)."""
    paths = _summary_path(path)
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    models = {(s.get("model") or {}).get("id") for s in summaries}
    if len(models) > 1:
        names = ", ".join(p.parent.name for p in paths)
        raise ValueError(f"{path} holds traces of different models ({names}); open one trace folder")
    merged = summaries[0] if len(summaries) == 1 else _merge(summaries)
    return ExpertMap(merged, folder=paths[0].parent if len(paths) == 1 else None)


def _merge(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Add up traces of the same model (e.g. one per rank): topics are matched by name."""
    base = dict(summaries[0])
    topics: list[str] = []
    for s in summaries:
        topics += [t for t in s.get("topics") or [] if t not in topics]
    layers = len(base.get("layers") or [])
    width = max(len(layer) for s in summaries for topic in s.get("counts") or [] for layer in topic)
    counts = [[[0.0] * width for _ in range(layers)] for _ in topics]
    gates = [[[0.0] * width for _ in range(layers)] for _ in topics]
    tokens = [[0.0] * layers for _ in topics]
    for s in summaries:
        for ti, topic in enumerate(s.get("topics") or []):
            t = topics.index(topic)
            for li in range(layers):
                tokens[t][li] += s["tokens"][ti][li]
                for e, v in enumerate(s["counts"][ti][li]):
                    counts[t][li][e] += v
                for e, v in enumerate(s["gates"][ti][li]):
                    gates[t][li][e] += v
    base.update(
        topics=topics,
        counts=counts,
        gates=gates,
        tokens=tokens,
        forwards=sum(int(s.get("forwards") or 0) for s in summaries),
        name="+".join(str(s.get("name")) for s in summaries),
    )
    return base


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def _descending(value: float | None) -> float:
    """Sort key: highest first, ``None`` last."""
    return -value if value is not None else math.inf


def _take(shares: Sequence[float], coverage: float) -> tuple[list[int], float]:
    """The fewest experts (highest share first) whose shares add up to ``coverage``."""
    order = sorted(range(len(shares)), key=lambda e: (-shares[e], e))
    kept, total = [], 0.0
    for e in order:
        if total >= coverage - 1e-12 or shares[e] <= 0:
            break
        kept.append(e)
        total += shares[e]
    return sorted(kept), total


@dataclass
class Selection:
    """The experts to keep per layer for one topic (derived from measured routing)."""

    topic: str
    by: str
    coverage: float
    model: str
    tokens: int
    layers: list[dict[str, Any]]
    """Per MoE layer: ``layer``, ``block``, ``experts_module``, ``router_module``, ``shared_module``,
    ``keep`` (expert ids), ``covered`` (share of the topic's routing they take) and ``of`` (experts)."""
    routing: dict[str, Any] = field(default_factory=dict)
    run: str | None = None

    def keep(self, layer: int) -> list[int]:
        for row in self.layers:
            if row["layer"] == layer:
                return list(row["keep"])
        raise KeyError(f"Layer {layer} is not an MoE layer of this trace")

    @property
    def kept(self) -> int:
        return sum(len(row["keep"]) for row in self.layers)

    @property
    def total(self) -> int:
        return sum(int(row["of"]) for row in self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "innards.experts.selection/1.0",
            "model": self.model,
            "run": self.run,
            "topic": self.topic,
            "derived": (
                f"Derived from the measured routing of {self.tokens:,} {self.topic!r} tokens: per layer, the fewest "
                f"experts that take {self.coverage:.0%} of the topic's routing by {self.by}."
            ),
            "method": {"by": self.by, "coverage": self.coverage},
            "routing": dict(self.routing),
            "summary": {
                "tokens": self.tokens,
                "kept": self.kept,
                "of": self.total,
                "share_kept": round(self.kept / self.total, 4) if self.total else None,
            },
            "layers": [dict(row) for row in self.layers],
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    def __str__(self) -> str:
        share = f"{self.kept / self.total:.0%}" if self.total else "n/a"
        lines = [
            f"{self.topic}: keep {self.kept} of {self.total} routed experts ({share}) for {self.coverage:.0%} "
            f"coverage by {self.by}, from {self.tokens:,} tokens (derived from measured routing)"
        ]
        for row in self.layers:
            lines.append(
                f"  layer {row['layer']:>3}: {len(row['keep']):>3}/{row['of']} experts cover "
                f"{row['covered']:.0%}  {row['keep']}"
            )
        return "\n".join(lines)


class ExpertMap:
    """Topic x layer x expert routing from one trace (or several of the same model, added up)."""

    def __init__(self, summary: Mapping[str, Any], folder: str | Path | None = None) -> None:
        self.data = dict(summary)
        self.folder = Path(folder) if folder is not None else None
        self.model: str = str((self.data.get("model") or {}).get("id") or "")
        self.routing: dict[str, Any] = dict((self.data.get("model") or {}).get("routing") or {})
        self.layer_info: list[dict[str, Any]] = list(self.data.get("layers") or [])
        self.layers: list[int] = [int(m["layer"]) for m in self.layer_info]
        self.topics: list[str] = list(self.data.get("topics") or [])
        self._counts: list[list[list[float]]] = self.data.get("counts") or []
        self._gates: list[list[list[float]]] = self.data.get("gates") or []
        self._tokens: list[list[float]] = self.data.get("tokens") or []

    # ------------------------------------------------------------------ measured

    def _t(self, topic: str) -> int:
        try:
            return self.topics.index(topic)
        except ValueError:
            raise KeyError(f"No topic {topic!r}; measured: {', '.join(self.topics) or 'none'}") from None

    def _l(self, layer: int) -> int:
        try:
            return self.layers.index(layer)
        except ValueError:
            raise KeyError(f"Layer {layer} is not an MoE layer; MoE layers: {self.layers}") from None

    def experts(self, layer: int) -> int:
        """Routed experts in ``layer``."""
        return int(self.layer_info[self._l(layer)]["experts"])

    def tokens(self, topic: str, layer: int | None = None) -> int:
        """Tokens of ``topic`` measured (in ``layer``, or the most any layer saw)."""
        row = self._tokens[self._t(topic)]
        return int(row[self._l(layer)] if layer is not None else max(row, default=0))

    def counts(self, topic: str) -> list[list[int]]:
        """``[layer][expert]``: how often each expert was in a token's top-k (measured)."""
        t = self._t(topic)
        rows = zip(self.layers, self._counts[t], strict=True)
        return [[int(v) for v in row[: self.experts(layer)]] for layer, row in rows]

    @property
    def weights(self) -> dict[int, str | None]:
        """Per layer, what the weights are (``applied``, ``gather`` or ``topk_values``; see routing)."""
        return {int(m["layer"]): m.get("weights") for m in self.layer_info}

    # ------------------------------------------------------------------ derived

    def selection(self, topic: str) -> list[list[float | None]]:
        """``[layer][expert]``: share of the topic's tokens whose top-k included the expert (derived)."""
        t = self._t(topic)
        return [
            [_ratio(v, self._tokens[t][li]) for v in row[: self.experts(layer)]]
            for li, (layer, row) in enumerate(zip(self.layers, self._counts[t], strict=True))
        ]

    def gate(self, topic: str) -> list[list[float | None]]:
        """``[layer][expert]``: the expert's average share of a token's routing weight (derived)."""
        t = self._t(topic)
        return [
            [_ratio(v, self._tokens[t][li]) for v in row[: self.experts(layer)]]
            for li, (layer, row) in enumerate(zip(self.layers, self._gates[t], strict=True))
        ]

    def lift(self, topic: str) -> list[list[float | None]]:
        """``[layer][expert]``: selection here / selection for all other topics together (derived).

        ``None`` where the other topics have no tokens or never chose the expert.
        """
        t = self._t(topic)
        out = []
        for li, layer in enumerate(self.layers):
            width = self.experts(layer)
            rest_tokens = sum(self._tokens[o][li] for o in range(len(self.topics)) if o != t)
            row = []
            for e in range(width):
                here = _ratio(self._counts[t][li][e], self._tokens[t][li])
                rest_count = sum(self._counts[o][li][e] for o in range(len(self.topics)) if o != t)
                rest = _ratio(rest_count, rest_tokens)
                row.append(here / rest if here is not None and rest else None)
            out.append(row)
        return out

    def metric(self, topic: str, by: str = "selection") -> list[list[float | None]]:
        if by not in METRICS:
            raise ValueError(f"by must be one of {', '.join(METRICS)}")
        fn = {"selection": self.selection, "gate": self.gate, "lift": self.lift}[by]
        return fn(topic)

    def shares(self, topic: str, layer: int, by: str = "selection") -> list[float]:
        """Each expert's share of the topic's routing in ``layer`` (sums to 1; derived)."""
        if by not in ("selection", "gate"):
            raise ValueError("Shares are by selection or by gate")
        t, li = self._t(topic), self._l(layer)
        row = (self._counts if by == "selection" else self._gates)[t][li][: self.experts(layer)]
        total = sum(row)
        return [v / total if total else 0.0 for v in row]

    def needed(self, topic: str, coverage: float = 0.9, by: str = "selection") -> list[int]:
        """Per layer, how many experts it takes to cover ``coverage`` of the topic's routing (derived)."""
        return [len(_take(self.shares(topic, layer, by), coverage)[0]) for layer in self.layers]

    def top(self, topic: str, layer: int, n: int = 8, by: str = "selection") -> list[dict[str, Any]]:
        """The ``n`` experts with the highest ``by`` for ``topic`` in ``layer``, with selection, gate and lift."""
        li = self._l(layer)
        sel, gate, lift = self.selection(topic)[li], self.gate(topic)[li], self.lift(topic)[li]
        key = {"selection": sel, "gate": gate, "lift": lift}[by]
        order = sorted(range(len(sel)), key=lambda e: (_descending(key[e]), e))
        return [{"layer": layer, "expert": e, "selection": sel[e], "gate": gate[e], "lift": lift[e]} for e in order[:n]]

    def specialists(self, topic: str, n: int = 10, min_selection: float = 0.05) -> list[dict[str, Any]]:
        """Experts chosen far more for this topic than for the others (highest lift, derived).

        Only experts that at least ``min_selection`` of the topic's tokens chose are considered;
        an expert the other topics never chose ranks first (``"only": True``).
        """
        rows: list[dict[str, Any]] = []
        for li, layer in enumerate(self.layers):
            sel, lift = self.selection(topic)[li], self.lift(topic)[li]
            t = self._t(topic)
            rest = sum(self._tokens[o][li] for o in range(len(self.topics)) if o != t)
            for e, s in enumerate(sel):
                if s is None or s < min_selection or not rest:
                    continue
                only = lift[e] is None
                rows.append({"layer": layer, "expert": e, "selection": s, "lift": lift[e], "only": only})
        rows.sort(key=lambda r: (not r["only"], -(r["lift"] or 0), -r["selection"]))
        return rows[:n]

    def select(self, topic: str, coverage: float = 0.9, by: str = "selection") -> Selection:
        """Per layer, the fewest experts that take ``coverage`` of the topic's routing (derived)."""
        if not 0 < coverage <= 1:
            raise ValueError("coverage is a share of the routing, in (0, 1]")
        rows = []
        for info in self.layer_info:
            layer = int(info["layer"])
            kept, covered = _take(self.shares(topic, layer, by), coverage)
            rows.append(
                {
                    "layer": layer,
                    "block": info.get("block"),
                    "experts_module": info.get("experts_module"),
                    "router_module": info.get("router_module"),
                    "shared_module": info.get("shared_module"),
                    "keep": kept,
                    "covered": round(covered, 4),
                    "of": int(info["experts"]),
                }
            )
        return Selection(
            topic=topic,
            by=by,
            coverage=coverage,
            model=self.model,
            tokens=self.tokens(topic),
            layers=rows,
            routing=dict(self.routing),
            run=self.data.get("run"),
        )

    def overlap(self, a: str, b: str, coverage: float = 0.9, by: str = "selection") -> list[float | None]:
        """Per layer, the Jaccard overlap of two topics' expert sets at ``coverage`` (derived)."""
        sa, sb = self.select(a, coverage, by), self.select(b, coverage, by)
        out: list[float | None] = []
        for ra, rb in zip(sa.layers, sb.layers, strict=True):
            x, y = set(ra["keep"]), set(rb["keep"])
            out.append(len(x & y) / len(x | y) if x | y else None)
        return out

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def __repr__(self) -> str:
        tokens = ", ".join(f"{t}: {self.tokens(t):,}" for t in self.topics)
        return f"<ExpertMap {self.model!r}: {len(self.layers)} MoE layers; tokens {tokens or 'none'}>"


class TokenRouting:
    """Every recorded token's chosen experts, from a trace folder's token shards (needs torch)."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        summary = json.loads((self.folder / "summary.json").read_text(encoding="utf-8"))
        self.layers: list[int] = [int(m["layer"]) for m in summary.get("layers") or []]
        self.experts: list[int] = [int(m["experts"]) for m in summary.get("layers") or []]
        self.topics: list[str] = list(summary.get("topics") or [])
        self.shards: list[str] = list(summary.get("shards") or [])
        vocab = self.folder / "vocab.json"
        self.vocab: dict[int, str] = (
            {int(k): v for k, v in json.loads(vocab.read_text(encoding="utf-8")).items()} if vocab.is_file() else {}
        )
        self._data: dict[str, Any] | None = None

    def _all(self) -> dict[str, Any]:
        import torch
        from safetensors.torch import load_file

        if self._data is None:
            parts = [load_file(str(self.folder / s)) for s in self.shards]
            keys = ("experts", "weights", "token", "sequence", "position", "topic")
            if parts:
                self._data = {k: torch.cat([p[k] for p in parts]) for k in keys}
            else:
                self._data = {k: torch.zeros(0) for k in keys}
        return self._data

    def text(self, token_id: int) -> str:
        return self.vocab.get(int(token_id), str(int(token_id)))

    def __len__(self) -> int:
        return int(self._all()["token"].numel())

    def sequences(self, topic: str | None = None) -> list[dict[str, Any]]:
        """Recorded sequences: ``{"sequence", "topic", "tokens", "preview"}``, in recording order."""
        import torch

        d = self._all()
        if not d["sequence"].numel():
            return []
        ids, counts = torch.unique(d["sequence"], return_counts=True)
        out = []
        for seq, count in zip(ids.tolist(), counts.tolist(), strict=True):
            rows = (d["sequence"] == seq).nonzero().reshape(-1)
            label = self.topics[int(d["topic"][rows[0]])] if self.topics else None
            if topic is not None and label != topic:
                continue
            order = rows[torch.argsort(d["position"][rows], stable=True)]
            preview = "".join(self.text(t) for t in d["token"][order[:24]].tolist())
            out.append({"sequence": seq, "topic": label, "tokens": count, "preview": preview})
        return out

    def sequence(self, sequence: int) -> dict[str, Any]:
        """One sequence in token order: ids, text, and per token ``[layer][slot]`` experts and weights."""
        import torch

        d = self._all()
        rows = (d["sequence"] == sequence).nonzero().reshape(-1)
        if not rows.numel():
            raise KeyError(f"No sequence {sequence} in {self.folder}")
        rows = rows[torch.argsort(d["position"][rows], stable=True)]
        ids = d["token"][rows].tolist()
        weights = d["weights"][rows]
        return {
            "sequence": sequence,
            "topic": self.topics[int(d["topic"][rows[0]])] if self.topics else None,
            "layers": self.layers,
            "ids": ids,
            "text": [self.text(t) for t in ids],
            "positions": d["position"][rows].tolist(),
            "experts": d["experts"][rows].tolist(),
            "weights": [[[None if math.isnan(v) else v for v in slot] for slot in tok] for tok in weights.tolist()],
        }

    def _ids(self, token: str | int) -> list[int]:
        if isinstance(token, int):
            return [token]
        exact = [i for i, t in self.vocab.items() if t == token]
        return exact or [i for i, t in self.vocab.items() if t.strip() == token.strip()]

    def token(self, token: str | int, topic: str | None = None) -> dict[str, Any]:
        """Where a token goes: per layer, how often each expert was in its top-k (measured counts)."""
        import torch

        d = self._all()
        ids = self._ids(token)
        mask = torch.isin(d["token"], torch.tensor(ids, dtype=d["token"].dtype)) if ids else d["token"] < -1
        if topic is not None:
            mask &= d["topic"] == self.topics.index(topic)
        experts = d["experts"][mask].long()  # [n, layers, k]
        counts = []
        for li, width in enumerate(self.experts):
            chosen = experts[:, li, :].reshape(-1)
            chosen = chosen[(chosen >= 0) & (chosen < width)]
            counts.append(torch.bincount(chosen, minlength=width).tolist())
        return {"token": token, "ids": ids, "occurrences": int(mask.sum()), "layers": self.layers, "counts": counts}

    def expert(self, layer: int, expert: int, n: int = 20, topic: str | None = None) -> list[dict[str, Any]]:
        """The tokens most often routed to ``expert`` in ``layer``: count, occurrences and their share."""
        import torch

        d = self._all()
        li = self.layers.index(layer)
        keep = torch.ones_like(d["token"], dtype=torch.bool)
        if topic is not None:
            keep &= d["topic"] == self.topics.index(topic)
        tokens = d["token"][keep].long()
        hit = (d["experts"][keep][:, li, :] == expert).any(-1)
        if not tokens.numel():
            return []
        size = int(tokens.max()) + 1
        seen = torch.bincount(tokens.clamp_min(0), minlength=size)
        routed = torch.bincount(tokens[hit].clamp_min(0), minlength=size)
        order = torch.argsort(routed, descending=True, stable=True)[:n]
        return [
            {
                "id": int(i),
                "text": self.text(int(i)),
                "routed": int(routed[i]),
                "occurrences": int(seen[i]),
                "share": int(routed[i]) / int(seen[i]) if int(seen[i]) else None,
            }
            for i in order.tolist()
            if int(routed[i]) > 0
        ]
