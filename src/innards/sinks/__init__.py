"""Sinks receive finished TurnRecords in a background thread.

A sink is anything with ``write(record)`` and ``close()``. Pass sinks to ``observe`` as
spec strings (``"console"``), sink objects, or plain callables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from innards.schema import TurnRecord

__all__ = [
    "CallbackSink",
    "ConsoleSink",
    "Sink",
    "SinkSpec",
    "available_sinks",
    "make_sink",
    "make_sinks",
    "register_sink",
]


@runtime_checkable
class Sink(Protocol):
    def write(self, record: TurnRecord) -> None: ...

    def close(self) -> None: ...


SinkSpec = str | Sink | Callable[["TurnRecord"], None]

_REGISTRY: dict[str, Callable[[str], Sink]] = {}


def register_sink(scheme: str, factory: Callable[[str], Sink]) -> None:
    """Register a factory for ``"<scheme>"`` or ``"<scheme>:<arg>"`` spec strings."""
    _REGISTRY[scheme] = factory


def available_sinks() -> list[str]:
    return sorted(_REGISTRY)


class CallbackSink:
    """Wrap a callable as a sink."""

    def __init__(self, callback: Callable[[TurnRecord], None]) -> None:
        self._callback = callback

    def write(self, record: TurnRecord) -> None:
        self._callback(record)

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"CallbackSink({self._callback!r})"


def make_sink(spec: SinkSpec) -> Sink:
    """Create a sink from a spec string, a sink object or a callable."""
    if isinstance(spec, str):
        scheme = spec.split("://", 1)[0] if "://" in spec else spec.split(":", 1)[0]
        factory = _REGISTRY.get(scheme)
        if factory is None:
            raise ValueError(f"Unknown sink {spec!r}. Available sinks: {', '.join(available_sinks())}")
        return factory(spec)
    if isinstance(spec, Sink):
        return spec
    if callable(spec):
        return CallbackSink(spec)
    raise TypeError(f"Cannot use {spec!r} as a sink")


def make_sinks(specs: Iterable[SinkSpec] | None) -> list[Sink]:
    return [make_sink(spec) for spec in (specs or ())]


from innards.sinks.console import ConsoleSink  # noqa: E402  (after Sink is defined)

register_sink("console", ConsoleSink.from_spec)
