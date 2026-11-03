"""What the view needs from a runner: the code that loads models and runs conversations.

Innards never loads or runs a model. The view's Load, Send and Run set controls call a runner that
the host application provides (``innards-exp serve`` provides one for the lineup's original
checkpoints). Without a runner the view monitors whatever your own code generates.

A runner streams each turn to the page as server-sent events through a :class:`TurnJob`:
``start``, then per turn ``turn_start``, ``token`` ..., ``turn`` (the finished record) or ``error``,
and finally ``end``.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

__all__ = ["Runner", "RunnerError", "TurnJob"]


class RunnerError(Exception):
    """A request the runner cannot serve in its current state (HTTP 409 unless ``status`` says otherwise)."""

    def __init__(self, message: str, status: int = 409, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


class TurnJob:
    """One chat turn, or a conversation set's turns, in flight: its events and what has been generated."""

    def __init__(self, kind: str, label: str, total: int = 1) -> None:
        self.kind = kind
        self.label = label
        self.total = total
        self.cancel = threading.Event()
        self._events: queue.SimpleQueue[tuple[str, Any] | None] = queue.SimpleQueue()
        self._text: list[str] = []
        self._current: dict[str, Any] = {}
        self._started = time.perf_counter()
        self.done = False

    def _elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 1)

    def turn_start(self, turn: int, message: str, step: int | None = None) -> None:
        self._text = []
        self._current = {"turn": turn, "message": message, "step": step, "of": self.total}
        self.put("turn_start", self._current)

    def token(self, text: str, generated: int | None = None) -> None:
        self._text.append(text)
        self.put("token", {"text": text, "generated": generated, "elapsed_ms": self._elapsed_ms()})

    def put(self, event: str, data: Any) -> None:
        self._events.put((event, data))

    def close(self) -> None:
        self.done = True
        self._events.put(None)

    def snapshot(self) -> dict[str, Any]:
        """What a page that reconnects mid-turn needs to show the turn in progress."""
        return {
            "kind": self.kind,
            "label": self.label,
            **self._current,
            "text": "".join(self._text),
            "elapsed_ms": self._elapsed_ms(),
        }

    def sse(self, heartbeat_s: float = 10.0) -> Iterator[str]:
        """Server-sent events until the job ends. Comments keep idle connections open during prefill."""
        yield _sse("start", {"kind": self.kind, "label": self.label, "total": self.total})
        while True:
            try:
                item = self._events.get(timeout=heartbeat_s)
            except queue.Empty:
                yield ": working\n\n"
                continue
            if item is None:
                yield _sse("end", {"kind": self.kind})
                return
            yield _sse(*item)


@runtime_checkable
class Runner(Protocol):
    """Loads original checkpoints and runs conversations with ``innards.observe`` attached."""

    name: str

    @property
    def watch(self) -> Any:
        """The :class:`~innards.Watch` on the loaded model, or ``None``."""
        ...

    def status(self) -> dict[str, Any]:
        """``{"state": idle | validating | loading | ready | generating | error, "model", "error", ...}``."""
        ...

    def models(self) -> dict[str, Any]:
        """What the model picker lists: ``{"lineup": [...], "downloaded": [...]}``."""
        ...

    def validate(
        self, model: str, precision: str | None, conversation_set: str | None, document: str | None = None
    ) -> dict[str, Any]:
        """The before-download checks (:func:`innards.validate.validate`) for this machine."""
        ...

    def load(
        self,
        model: str,
        precision: str | None,
        device: str | None,
        conversation_set: str | None = None,
        document: str | None = None,
    ) -> dict[str, Any]:
        """Validate (sizing the KV for ``conversation_set``), then load in the background.

        Raises :class:`RunnerError` with ``detail={"report": ...}`` when a check blocks.
        """
        ...

    def unload(self) -> dict[str, Any]: ...

    def new_conversation(self, conversation_set: str | None = None) -> dict[str, Any]: ...

    def chat(self, message: str) -> TurnJob: ...

    def run_set(
        self,
        conversation_set: str,
        document: str | None = None,
        document_name: str | None = None,
        steps: int | None = None,
    ) -> TurnJob: ...

    def stop(self) -> bool: ...

    def count_tokens(self, text: str) -> int | None:
        """Tokens ``text`` takes with the loaded model's tokenizer (``None`` without a model)."""
        ...
