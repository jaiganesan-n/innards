"""What the view needs from a runner: the code that sends the Conversation tab's messages to a model.

Package use: :class:`innards.view.chat.ChatRunner` (``trace.view(chat=True)``) talks to the model your
code loaded and traced. Independent use: :class:`innards.view.loader.ModelRunner` (``innards view``) loads
the MoE checkpoint picked on the page, then talks to it the same way. Without a runner the page shows
traces and stored conversations only.

A runner streams each turn to the page as server-sent events through a :class:`TurnJob`:
``start``, ``turn_start``, ``token`` ..., ``turn`` (the finished turn) or ``error``, and finally ``end``.
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
    """One chat turn in flight: its events and what has been generated."""

    def __init__(self, kind: str, label: str) -> None:
        self.kind = kind
        self.label = label
        self.cancel = threading.Event()
        self._events: queue.SimpleQueue[tuple[str, Any] | None] = queue.SimpleQueue()
        self._text: list[str] = []
        self._current: dict[str, Any] = {}
        self._started = time.perf_counter()
        self.done = False

    def _elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 1)

    def turn_start(self, turn: int, message: str, topic: str | None = None) -> None:
        self._text = []
        self._current = {"turn": turn, "message": message, "topic": topic}
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
        yield _sse("start", {"kind": self.kind, "label": self.label})
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
    """Sends the Conversation tab's messages to a traced model and streams the replies."""

    name: str

    def status(self) -> dict[str, Any]:
        """``{"state": ready | generating | error, "model", "trace": {"run", "name"}, "conversation", ...}``."""
        ...

    def new_conversation(self) -> dict[str, Any]:
        """Start a new conversation (an empty history); returns :meth:`status`."""
        ...

    def chat(self, message: str, topic: str | None = None) -> TurnJob:
        """Send ``message`` as the next user turn, recorded under ``topic``; the reply streams through the job."""
        ...

    def stop(self) -> bool:
        """Stop the reply being generated (``False`` when none is)."""
        ...
