"""Non-blocking observation: the generation path only timestamps, counts and enqueues.

* :func:`observe` / :class:`Watch`: hooks on a model you already loaded, one turn per ``generate()``.
* :class:`BoundedEventQueue` never blocks; when full it drops the event and counts it.
* :class:`Observer` workers build TurnRecords, run the feature analyzers and write to sinks.
* :class:`ProbeRecorder`: the timed, fail-open hooks of feature probes and their non-blocking copies.
* Fail-open: observer, analyzer and sink errors are logged and counted; generation continues.
"""

from innards.observer.events import Event, EventKind, TurnEnd, TurnKey, TurnStart, TurnStats
from innards.observer.observer import Observer
from innards.observer.probes import CapturedTensors, ProbeRecorder
from innards.observer.queue import BoundedEventQueue
from innards.observer.watch import CapturedTurn, Watch, observe

__all__ = [
    "BoundedEventQueue",
    "CapturedTensors",
    "CapturedTurn",
    "Event",
    "EventKind",
    "Observer",
    "ProbeRecorder",
    "TurnEnd",
    "TurnKey",
    "TurnStart",
    "TurnStats",
    "Watch",
    "observe",
]
