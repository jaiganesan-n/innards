"""Incremental detokenization for streaming, with stop-string hold-back."""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = ["IncrementalDecoder", "strip_stop"]


def strip_stop(text: str, stop_strings: Sequence[str]) -> tuple[str, bool]:
    """Cut ``text`` at the first stop string. Returns ``(text, stopped)``."""
    cut = min((i for i in (text.find(s) for s in stop_strings if s) if i >= 0), default=-1)
    return (text[:cut], True) if cut >= 0 else (text, False)


class IncrementalDecoder:
    """Turn a growing list of token ids into text deltas.

    Decodes a short window (prefix/read offsets) instead of the whole sequence,
    holds back incomplete UTF-8 sequences, and holds back any tail that could be
    the start of a stop string.
    """

    def __init__(self, decode: Callable[[list[int]], str], stop_strings: Sequence[str] = ()) -> None:
        self._decode = decode
        self._stop = [s for s in stop_strings if s]
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0
        self._decoded = ""  # all text decoded so far
        self._released = 0  # characters of _decoded handed to the caller
        self.stopped = False

    @property
    def text(self) -> str:
        """Full text so far, cut at the first stop string."""
        return strip_stop(self._decoded, self._stop)[0]

    def _hold(self, text: str) -> int:
        """Length of the longest suffix of ``text`` that is a proper prefix of a stop string."""
        best = 0
        for stop in self._stop:
            for n in range(min(len(stop) - 1, len(text)), 0, -1):
                if text.endswith(stop[:n]):
                    best = max(best, n)
                    break
        return best

    def push(self, ids: Sequence[int]) -> str:
        if self.stopped:
            return ""
        self._ids.extend(ids)
        prefix_text = self._decode(self._ids[self._prefix : self._read])
        new_text = self._decode(self._ids[self._prefix :])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._decoded += new_text[len(prefix_text) :]
            self._prefix, self._read = self._read, len(self._ids)
        return self._release(final=False)

    def flush(self) -> str:
        if not self.stopped and self._read < len(self._ids):
            tail = self._decode(self._ids[self._prefix :])[len(self._decode(self._ids[self._prefix : self._read])) :]
            self._decoded += tail
            self._read = len(self._ids)
        return self._release(final=True)

    def _release(self, final: bool) -> str:
        text, stopped = strip_stop(self._decoded, self._stop)
        if stopped:
            self.stopped = True
            end = len(text)
        else:
            end = len(text) if final else len(text) - self._hold(text)
        out = text[self._released : end] if end > self._released else ""
        self._released = max(self._released, end)
        return out
