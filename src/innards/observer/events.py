"""Small events passed from the generation path to observer workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from innards.schema import TurnRecord

__all__ = ["Event", "EventKind", "TurnEnd", "TurnKey", "TurnStart", "TurnStats"]

TurnKey = tuple[str, int]
"""``(conversation id, turn)``."""


class EventKind(IntEnum):
    TURN_START = 1
    TOKEN = 2
    TURN_END = 3
    STOP = 9


class Event(NamedTuple):
    kind: int
    t_ns: int
    key: TurnKey | None
    data: Any = None


@dataclass(slots=True)
class TurnStart:
    """What is known before generation begins. Built outside the token loop."""

    seed: dict[str, Any]
    """Partial TurnRecord fields known up front (conversation, turn, prompt shape, message ...)."""
    t_request_ns: int
    builder: Callable[[TurnStart, TurnEnd, TurnStats], TurnRecord]
    """Turns raw data into a TurnRecord. Runs in an observer worker, never in generation."""


@dataclass(slots=True)
class TurnEnd:
    """Raw facts captured right after generation returns (metadata only, no device sync)."""

    generated_tokens: int
    text: str
    finish_reason: str | None
    cache_tokens: int | None
    t_gen_start_ns: int | None
    t_first_token_ns: int | None
    t_last_token_ns: int | None
    t_end_ns: int
    cache_snapshot: Any = None
    """Backend-specific cache metadata (shapes and dtypes), turned into bytes by a worker."""
    counters: dict[str, int] = field(default_factory=dict)
    """Cheap engine counters read at turn end (e.g. CUDA allocator stats)."""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Producer-specific facts for the record builder (``observe`` passes token ids and probe captures)."""


@dataclass(slots=True)
class TurnStats:
    """Accumulated by the observer worker from TOKEN events."""

    tokens_seen: int = 0
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    peak_sampled_bytes: int | None = None
    dropped_events: int = 0
    errors: int = 0
