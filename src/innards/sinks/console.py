"""Console sink: a compact per-turn summary (or JSON lines) on stderr/stdout."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from innards.schema import TurnRecord

__all__ = ["ConsoleSink", "format_bytes", "format_turn"]


def format_bytes(n: float | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover


def _num(value: float | None, fmt: str = "{:.0f}", suffix: str = "") -> str:
    return "n/a" if value is None else fmt.format(value) + suffix


def format_turn(record: TurnRecord) -> str:
    """Three-line human summary of a TurnRecord."""
    m, t, kv, mem, tm = record.model, record.tokens, record.kv, record.memory, record.timing
    arch = m.architecture or "?"
    if m.attention and m.attention != arch:
        arch = f"{arch}/{m.attention}"
    reuse = "" if kv.prefix_reused is None else (" reused" if kv.prefix_reused else " no reuse")
    line1 = (
        f"[innards] turn {record.turn} · {m.id} ({arch}) · prompt {_num(t.prompt)} "
        f"(cached {_num(t.cached)}{reuse}, recomputed {_num(t.recomputed)}) · "
        f"generated {_num(t.generated)} · context {_num(t.context_after)}"
    )
    groups = kv.by_layer_type or {}
    gap = ""
    if kv.measured_bytes is not None and kv.predicted_bytes:
        gap = f" ({(kv.measured_bytes - kv.predicted_bytes) / kv.predicted_bytes:+.1%})"
    line2 = (
        f"          KV measured {format_bytes(kv.measured_bytes)} vs predicted {format_bytes(kv.predicted_bytes)}"
        f"{gap} [{kv.measured_source or 'n/a'}] · global {format_bytes(groups.get('global'))} · "
        f"sliding {format_bytes(groups.get('sliding'))} · recurrent {format_bytes(groups.get('recurrent'))}"
    )
    peak_label = "peak RSS" if mem.source == "psutil_rss" else "peak"
    line3 = (
        f"          TTFT {_num(tm.ttft_ms, '{:.0f}', ' ms')} · prefill {_num(tm.prefill_ms, '{:.0f}', ' ms')} · "
        f"decode {_num(tm.decode_tokens_per_s, '{:.1f}', ' tok/s')} · total {_num(tm.total_ms, '{:.0f}', ' ms')} · "
        f"{peak_label} {format_bytes(mem.peak_bytes)} · dropped {record.observer.dropped_events}"
    )
    return "\n".join((line1, line2, line3))


class ConsoleSink:
    """Print each TurnRecord. ``fmt="text"`` (default) or ``"json"`` (one JSON object per line)."""

    def __init__(self, stream: TextIO | None = None, fmt: str = "text") -> None:
        if fmt not in {"text", "json"}:
            raise ValueError("fmt must be 'text' or 'json'")
        self._stream = stream
        self.fmt = fmt

    @classmethod
    def from_spec(cls, spec: str) -> ConsoleSink:
        """``console``, ``console:stdout``, ``console:stderr``, ``console:json`` or ``console:stdout:json``."""
        parts = spec.split(":")[1:]
        stream: TextIO | None = None
        fmt = "text"
        for part in parts:
            if part == "stdout":
                stream = sys.stdout
            elif part == "stderr":
                stream = sys.stderr
            elif part in {"json", "text"}:
                fmt = part
            elif part:
                raise ValueError(f"Unknown console sink option {part!r}")
        return cls(stream=stream, fmt=fmt)

    def write(self, record: TurnRecord) -> None:
        stream = self._stream or sys.stderr
        text = record.to_json() if self.fmt == "json" else format_turn(record)
        stream.write(text + "\n")
        stream.flush()

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"ConsoleSink(fmt={self.fmt!r})"
