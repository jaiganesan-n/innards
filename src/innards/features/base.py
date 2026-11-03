"""Building blocks for feature analyzers.

An analyzer covers one architectural feature (Gated DeltaNet, mHC, LatentMoE ...). It has three parts:

* :meth:`Analyzer.detect` reads the model config and says whether (and where) the feature exists.
* :meth:`Analyzer.attach` registers read-only forward hooks (the *probes*) through a :class:`Recorder`.
  Hooks only reduce small tensors on the device and hand them to the recorder, which copies them
  to the CPU without blocking. They never change a module's output.
* :meth:`Analyzer.analyze` runs in a worker thread after the turn, on CPU copies, and returns the
  turn's payload: ``{"title", "summary", "findings", "data"}``. :meth:`Analyzer.conversation` turns
  the payloads of a whole conversation into a section of the Analysis tab (findings and charts).

Charts are plain JSON specs (:func:`line`, :func:`heatmap`, :func:`bars`) that the web view draws.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "Analyzer",
    "Detection",
    "Payload",
    "Recorder",
    "TurnData",
    "bars",
    "fmt_bytes",
    "heatmap",
    "line",
    "rnd",
    "section",
    "text_config",
]

log = logging.getLogger("innards.features")

Payload = dict[str, Any]
"""``{"title": str, "summary": {name: number}, "findings": [str], "data": {...}}``."""

_NESTED = ("text_config", "language_config", "llm_config")


def text_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The decoder config (nested ``text_config`` for multimodal checkpoints)."""
    for key in _NESTED:
        sub = config.get(key)
        if isinstance(sub, Mapping) and sub:
            return dict(sub)
    return dict(config)


def rnd(value: float | None, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000 or unit == "TB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1000
    return f"{value:,.1f} TB"  # pragma: no cover


# --------------------------------------------------------------------------- charts


def line(
    title: str,
    x: Sequence[Any],
    series: Sequence[Mapping[str, Any]],
    *,
    x_label: str = "turn",
    y_label: str = "",
    unit: str | None = None,
    mode: str = "lines+markers",
) -> dict[str, Any]:
    """A line chart. Each series: ``{"name", "values", "model"?, "dash"?}``; ``model`` picks its fixed color."""
    return {
        "type": "line",
        "title": title,
        "x": list(x),
        "series": [dict(s) for s in series],
        "x_label": x_label,
        "y_label": y_label,
        "unit": unit,
        "mode": mode,
    }


def bars(
    title: str,
    x: Sequence[Any],
    series: Sequence[Mapping[str, Any]],
    *,
    x_label: str = "",
    y_label: str = "",
    unit: str | None = None,
    stacked: bool = False,
) -> dict[str, Any]:
    return {
        "type": "bar",
        "title": title,
        "x": list(x),
        "series": [dict(s) for s in series],
        "x_label": x_label,
        "y_label": y_label,
        "unit": unit,
        "stacked": stacked,
    }


def heatmap(
    title: str,
    x: Sequence[Any],
    y: Sequence[Any],
    z: Sequence[Sequence[float | None]],
    *,
    x_label: str = "turn",
    y_label: str = "layer",
    unit: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> dict[str, Any]:
    return {
        "type": "heatmap",
        "title": title,
        "x": list(x),
        "y": list(y),
        "z": [list(row) for row in z],
        "x_label": x_label,
        "y_label": y_label,
        "unit": unit,
        "zmin": zmin,
        "zmax": zmax,
    }


def section(
    feature: str,
    title: str,
    headline: str,
    findings: Sequence[str],
    charts: Sequence[Mapping[str, Any]] = (),
    *,
    kind: str = "feature",
) -> dict[str, Any]:
    """A section of the Analysis tab."""
    return {
        "id": feature,
        "kind": kind,
        "title": title,
        "headline": headline,
        "findings": [f for f in findings if f],
        "charts": [dict(c) for c in charts],
    }


# --------------------------------------------------------------------------- detection and data


@dataclass(frozen=True)
class Detection:
    """Where a feature lives in a model, read from its config."""

    feature: str
    layers: tuple[int, ...] = ()
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "layers": list(self.layers), "facts": dict(self.facts)}


class Recorder(Protocol):
    """What probes use on the generation path. Every call is constant time and never blocks."""

    def hook(
        self,
        feature: str,
        module: Any,
        fn: Callable[..., None],
        *,
        pre: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> None:
        """Register ``fn(module, inputs, output)`` as a timed, fail-open forward hook on ``module``.

        ``pre=True`` registers ``fn(module, inputs)`` (``fn(module, args, kwargs)`` with ``with_kwargs``)
        as a forward pre-hook; ``always_call`` runs a forward hook even when the forward raises.
        """
        ...

    def put(self, feature: str, name: str, layer: int, tensor: Any) -> None:
        """Hand a small tensor to the worker: a non-blocking copy to the CPU (pinned memory on CUDA)."""
        ...

    def at_turn_end(self, feature: str, fn: Callable[[Any], Any]) -> None:
        """Call ``fn(cache)`` on the generation thread once ``generate`` returns.

        Whatever ``fn`` returns (e.g. tensors it copied to the CPU with :meth:`to_cpu`) reaches the
        worker as ``TurnData.extras[feature]``.
        """
        ...

    def to_cpu(self, tensor: Any) -> Any:
        """A non-blocking CPU copy of ``tensor`` (pinned memory on CUDA), safe to read in the worker."""
        ...


@dataclass
class TurnData:
    """What a probe captured during one turn, on the CPU, plus the turn's light-tier record."""

    feature: str
    detection: Detection
    captures: dict[str, dict[int, list[tuple[int, Any]]]]
    """``name -> layer -> [(step, tensor)]``. Step 0 is the prefill; step k the k-th decoded token."""
    prefill_tokens: int
    """Tokens the prefill step processed (the new part of the prompt)."""
    token_ids: list[int]
    """Every token processed this turn, in order (prefill tokens, then generated ones)."""
    record: dict[str, Any]
    """The turn's TurnRecord (schema 1.0) without feature payloads."""
    config: dict[str, Any]
    cache_layers: list[dict[str, Any]] = field(default_factory=list)
    """Per cache layer: ``{"kind", "kv": [[shape, itemsize]], "state": [...], "extra": [[group, shape, itemsize]]}``."""
    state: dict[str, Any] = field(default_factory=dict)
    """Analyzer-owned memory carried from turn to turn within a conversation."""
    decode: Callable[[list[int]], str] | None = None
    """The tokenizer's ``decode``, when a tokenizer was given."""
    extras: dict[str, Any] = field(default_factory=dict)
    """Values handed over at turn end by :meth:`Recorder.at_turn_end` callbacks (CPU copies)."""

    def positions(self, step: int, length: int) -> list[int]:
        """Token positions (within this turn) covered by ``length`` rows captured at ``step``."""
        if step == 0:
            return list(range(max(self.prefill_tokens - length, 0), self.prefill_tokens))
        return [self.prefill_tokens + step - 1]

    def series(self, name: str) -> tuple[list[int], dict[int, Any]]:
        """Per layer, rows captured at every recorded step, concatenated in token order.

        Returns ``(positions, {layer: tensor[tokens, ...]})``; positions come from the first layer.
        """
        import torch

        rows = self.captures.get(name) or {}
        out: dict[int, Any] = {}
        positions: list[int] = []
        for i, layer in enumerate(sorted(rows)):
            items = sorted(rows[layer], key=lambda item: item[0])
            tensors = [t.reshape(-1, *t.shape[-1:]) if t.dim() >= 2 else t.reshape(-1, 1) for _, t in items]
            out[layer] = torch.cat(tensors, dim=0).float() if tensors else torch.empty(0)
            if i == 0:
                for (step, _), tensor in zip(items, tensors, strict=True):
                    positions.extend(self.positions(step, int(tensor.shape[0])))
        return positions, out

    def text_at(self, positions: Sequence[int]) -> list[str]:
        """Text of the tokens at ``positions`` (token ids as strings without a tokenizer)."""
        out = []
        for p in positions:
            if not 0 <= p < len(self.token_ids):
                out.append("")
            elif self.decode is None:
                out.append(str(self.token_ids[p]))
            else:
                try:
                    out.append(self.decode([self.token_ids[p]]))
                except Exception:
                    out.append(str(self.token_ids[p]))
        return out


class Analyzer:
    """Base class for one architectural feature.

    Subclasses set :attr:`name`, :attr:`title`, :attr:`reads` and :attr:`shows`, and implement
    :meth:`detect` (and :meth:`attach` if they need hooks), :meth:`analyze` and :meth:`conversation`.

    An analyzer instance is also callable, for explicit use on a captured turn::

        from innards import GatedDeltaNet
        GatedDeltaNet(model="Qwen/Qwen3.5-0.8B", args=watch.last_turn())
    """

    name: str = ""
    title: str = ""
    reads: str = ""
    """What it reads, e.g. "β and decay per token and layer"."""
    shows: str = ""
    """What it shows, e.g. "Write strength, retention, half-life"."""
    light: bool = False
    """``True`` when it needs no hooks (reads the cache layout and the light tier only)."""
    max_series: int = 400
    """Per-token series keep at most this many tokens per turn (the last ones)."""

    def detect(self, config: Mapping[str, Any]) -> Detection | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def attach(self, model: Any, detection: Detection, recorder: Recorder, layers: set[int] | None = None) -> int:
        """Register this feature's probes. Returns the number of hooks. Light analyzers register none."""
        return 0

    def analyze(self, turn: TurnData) -> Payload | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def conversation(
        self,
        turns: Sequence[Mapping[str, Any]],
        model: Mapping[str, Any],
        other: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """A section for the Analysis tab from the stored turns of one conversation (``None``: nothing to show).

        ``other`` is a second conversation ``(turns, model)`` to compare with, when the view asks for one.
        Chart series carry ``"model": <model id>`` so each model keeps one color everywhere.
        """
        return None

    # ------------------------------------------------------------------ explicit use

    def __call__(self, model: Any = None, args: Any = None) -> Payload | None:
        """Analyze a captured turn: ``args`` is :meth:`innards.Watch.last_turn` (or a list of them).

        ``model`` is the model id, config dict or config path the turn came from; it is only used to
        detect the feature when the capture did not already carry a detection.
        """
        from innards.features.registry import explicit_call

        return explicit_call(self, model, args)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}: {self.shows}>"

    # ------------------------------------------------------------------ helpers for subclasses

    @staticmethod
    def payload(title: str, summary: Mapping[str, Any], findings: Sequence[str], data: Mapping[str, Any]) -> Payload:
        return {"title": title, "summary": dict(summary), "findings": [f for f in findings if f], "data": dict(data)}

    @staticmethod
    def payloads(turns: Sequence[Mapping[str, Any]], feature: str) -> list[tuple[Mapping[str, Any], Payload]]:
        """``(turn, payload)`` for the turns that carry ``feature``."""
        out = []
        for turn in turns:
            payload = (turn.get("features") or {}).get(feature)
            if isinstance(payload, dict) and payload.get("data") is not None:
                out.append((turn, payload))
        return out
