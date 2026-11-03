"""A bounded, never-blocking event queue that drops and counts when full."""

from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

__all__ = ["BoundedEventQueue"]

_T = TypeVar("_T")


class BoundedEventQueue(Generic[_T]):
    """Multi-producer, single-consumer queue for the generation hot path.

    ``offer`` never waits: it checks the size, and either enqueues (a C-level
    ``SimpleQueue.put`` that cannot block) or drops the event and counts the drop.
    With several producers racing, the size can overshoot ``maxsize`` by at most
    the number of producers.
    """

    __slots__ = ("_dropped", "_lock", "_maxsize", "_q")

    def __init__(self, maxsize: int = 4096) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._q: queue.SimpleQueue[_T] = queue.SimpleQueue()
        self._maxsize = maxsize
        self._dropped = 0
        self._lock = threading.Lock()  # only taken on the (rare) drop path

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def dropped(self) -> int:
        return self._dropped

    def qsize(self) -> int:
        return self._q.qsize()

    def offer(self, item: _T) -> bool:
        """Enqueue ``item`` unless the queue is full. Never blocks. Returns ``False`` on drop."""
        if self._q.qsize() >= self._maxsize:
            with self._lock:
                self._dropped += 1
            return False
        self._q.put_nowait(item)
        return True

    def put_control(self, item: _T) -> None:
        """Enqueue a control item (e.g. a stop sentinel) regardless of the bound. Not for the hot path."""
        self._q.put_nowait(item)

    def get(self, timeout: float | None = None) -> _T | None:
        """Consumer side: wait up to ``timeout`` seconds for an item; ``None`` on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
