"""Background observer: builds TurnRecords off the generation path and fans them out to sinks."""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from innards.observer.events import Event, EventKind, TurnEnd, TurnKey, TurnStart, TurnStats
from innards.observer.queue import BoundedEventQueue
from innards.schema import TurnRecord
from innards.sinks import Sink

__all__ = ["Observer"]

log = logging.getLogger("innards.observe")

_STOP = Event(EventKind.STOP, 0, None, None)


@dataclass(slots=True)
class _TurnState:
    start: TurnStart | None
    stats: TurnStats = field(default_factory=TurnStats)


class Observer:
    """Consumes events from a bounded queue in a background thread.

    * :meth:`emit` is the only method the generation path calls. It takes a
      timestamp and offers a small event to the queue: constant time, never blocks,
      drops and counts when the queue is full.
    * A worker thread builds each :class:`~innards.schema.TurnRecord`.
    * A second thread writes records to sinks through its own bounded queue, so a
      slow sink cannot stall record building.
    * Fail-open: every observer or sink error is logged and counted; nothing is
      raised into generation.
    """

    def __init__(
        self,
        sinks: Iterable[Sink] = (),
        *,
        queue_size: int = 4096,
        sink_queue_size: int = 256,
        memory_sampler: Callable[[], int | None] | None = None,
        sample_interval_s: float = 0.05,
        name: str = "innards",
    ) -> None:
        self._events: BoundedEventQueue[Event] = BoundedEventQueue(queue_size)
        self._records: BoundedEventQueue[TurnRecord | None] = BoundedEventQueue(sink_queue_size)
        self._sinks = list(sinks)
        self._memory_sampler = memory_sampler
        self._sample_interval_ns = int(sample_interval_s * 1e9)
        self._last_sample_ns = 0
        self._states: dict[TurnKey, _TurnState] = {}
        self._waiters: dict[TurnKey, Future[TurnRecord]] = {}
        self._waiters_lock = threading.Lock()
        self._listeners: list[Callable[[TurnRecord], None]] = []
        self._errors = 0
        self._errors_lock = threading.Lock()
        self._processed = 0
        self._sunk = 0
        self._name = name
        self._worker: threading.Thread | None = None
        self._sink_worker: threading.Thread | None = None
        self._closed = False

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> Observer:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name=f"{self._name}-observer", daemon=True)
            self._sink_worker = threading.Thread(target=self._run_sinks, name=f"{self._name}-sinks", daemon=True)
            self._worker.start()
            self._sink_worker.start()
        return self

    def close(self, timeout: float = 5.0) -> None:
        """Drain queues, close sinks and stop the worker threads."""
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._events.put_control(_STOP)
            self._worker.join(timeout)
        if self._sink_worker is not None:
            self._records.put_control(None)
            self._sink_worker.join(timeout)
            if self._sink_worker.is_alive():
                log.warning("innards: sinks still writing after %.0fs; not closing them", timeout)
                return
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                self._count_error("closing sink %r", sink)
        with self._waiters_lock:
            for future in self._waiters.values():
                future.cancel()
            self._waiters.clear()

    # ----------------------------------------------------------------- stats

    @property
    def dropped(self) -> int:
        """Events and records dropped because a bounded queue was full."""
        return self._events.dropped + self._records.dropped

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def processed(self) -> int:
        """Records built so far."""
        return self._processed

    @property
    def sunk(self) -> int:
        """Records handed to every sink so far."""
        return self._sunk

    @property
    def sink_dropped(self) -> int:
        """Records dropped because the sink queue was full."""
        return self._records.dropped

    @property
    def sinks(self) -> list[Sink]:
        return list(self._sinks)

    # ----------------------------------------------------------------- producer side

    def emit(self, kind: int, key: TurnKey | None, data: Any = None) -> bool:
        """Hot path: timestamp and enqueue. Never blocks; returns ``False`` if the event was dropped."""
        return self._events.offer(Event(kind, time.perf_counter_ns(), key, data))

    def emit_control(self, kind: int, key: TurnKey | None, data: Any = None) -> None:
        """Turn start/end events. Never blocks and is never dropped.

        Called once before and once after generation (outside the token loop), so a turn
        always gets a record even when per-token events were dropped.
        """
        self._events.put_control(Event(kind, time.perf_counter_ns(), key, data))

    def expect(self, key: TurnKey) -> Future[TurnRecord]:
        """Register interest in a turn's record before the turn starts."""
        future: Future[TurnRecord] = Future()
        with self._waiters_lock:
            self._waiters[key] = future
        return future

    def forget(self, key: TurnKey) -> None:
        with self._waiters_lock:
            self._waiters.pop(key, None)

    def add_listener(self, callback: Callable[[TurnRecord], None]) -> None:
        """Call ``callback(record)`` in the observer thread for every finished record."""
        self._listeners.append(callback)

    # ----------------------------------------------------------------- worker side

    def _count_error(self, message: str, *args: object) -> None:
        with self._errors_lock:
            self._errors += 1
        log.exception("innards observer error: " + message, *args)

    def _run(self) -> None:
        while True:
            event = self._events.get()
            if event is None:
                continue
            if event.kind == EventKind.STOP:
                break
            try:
                self._handle(event)
            except Exception:
                self._count_error("handling %s", EventKind(event.kind).name)

    def _handle(self, event: Event) -> None:
        kind, key = event.kind, event.key
        if key is None:
            return
        if kind == EventKind.TOKEN:
            state = self._states.get(key)
            if state is None:
                state = self._states[key] = _TurnState(start=None)
            stats = state.stats
            stats.tokens_seen += int(event.data or 1)
            if stats.first_token_ns is None:
                stats.first_token_ns = event.t_ns
            stats.last_token_ns = event.t_ns
            self._maybe_sample(stats, event.t_ns)
        elif kind == EventKind.TURN_START:
            state = self._states.setdefault(key, _TurnState(start=None))
            state.start = event.data
            self._maybe_sample(state.stats, event.t_ns, force=True)
        elif kind == EventKind.TURN_END:
            state = self._states.pop(key, None) or _TurnState(start=None)
            self._maybe_sample(state.stats, event.t_ns, force=True)
            self._finish(key, state, event.data)

    def _maybe_sample(self, stats: TurnStats, now_ns: int, force: bool = False) -> None:
        if self._memory_sampler is None:
            return
        if not force and now_ns - self._last_sample_ns < self._sample_interval_ns:
            return
        self._last_sample_ns = now_ns
        try:
            value = self._memory_sampler()
        except Exception:
            self._count_error("sampling memory")
            return
        if value is not None and (stats.peak_sampled_bytes is None or value > stats.peak_sampled_bytes):
            stats.peak_sampled_bytes = value

    def _finish(self, key: TurnKey, state: _TurnState, end: TurnEnd) -> None:
        stats = state.stats
        stats.dropped_events = self.dropped
        stats.errors = self._errors
        record: TurnRecord | None = None
        if state.start is None:
            with self._errors_lock:
                self._errors += 1
            log.warning("innards: TURN_START for %s was dropped; recording a partial turn", key)
        else:
            try:
                record = state.start.builder(state.start, end, stats)
            except Exception:
                self._count_error("building record for %s", key)
        if record is None:
            record = _partial_record(key, state.start, end, stats)
        record.observer.dropped_events = self.dropped
        record.observer.errors = self._errors
        self._processed += 1
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception:
                self._count_error("listener %r", listener)
        with self._waiters_lock:
            future = self._waiters.pop(key, None)
        if future is not None and not future.done():
            future.set_result(record)
        self._records.offer(record)

    def _run_sinks(self) -> None:
        while True:
            record = self._records.get()
            if record is None:
                if self._closed:
                    break
                continue
            for sink in self._sinks:
                try:
                    sink.write(record)
                except Exception:
                    self._count_error("sink %r", sink)
            self._sunk += 1


def _partial_record(key: TurnKey, start: TurnStart | None, end: TurnEnd, stats: TurnStats) -> TurnRecord:
    """Fallback when the builder failed or its start event was dropped: keep what is certain."""
    from innards.schema import ModelInfo, OutputInfo, RuntimeInfo, TokenCounts

    seed = dict(start.seed) if start is not None else {}
    record = TurnRecord(
        conversation=key[0],
        turn=key[1],
        timestamp=str(seed.get("timestamp", "")),
        model=copy.copy(seed.get("model")) or ModelInfo(id="unknown"),
        runtime=copy.copy(seed.get("runtime")) or RuntimeInfo(backend="unknown"),
        run=seed.get("run"),
        message=seed.get("message"),
        set=seed.get("set"),
    )
    prompt = seed.get("prompt")
    record.tokens = TokenCounts(
        prompt=prompt,
        generated=end.generated_tokens,
        context_after=prompt + end.generated_tokens if prompt is not None else None,
        batch_size=seed.get("batch"),
    )
    record.output = OutputInfo(text=end.text, finish_reason=end.finish_reason)
    return record
