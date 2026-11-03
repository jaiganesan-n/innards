"""Timing from streamer timestamps (``time.perf_counter_ns``)."""

from __future__ import annotations

from innards.observer.events import TurnEnd
from innards.schema import TimingInfo

__all__ = ["timing_from"]


def _ms(start: int | None, end: int | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) / 1e6, 3)


def timing_from(t_request_ns: int | None, end: TurnEnd) -> TimingInfo:
    """TTFT and total time are measured from the request; prefill from the start of ``generate``.

    Decode speed counts tokens after the first one, over the time between the first
    and last token, so it excludes prefill.
    """
    decode: float | None = None
    first, last = end.t_first_token_ns, end.t_last_token_ns
    if first is not None and last is not None and last > first and end.generated_tokens > 1:
        decode = round((end.generated_tokens - 1) / ((last - first) / 1e9), 3)
    return TimingInfo(
        ttft_ms=_ms(t_request_ns, first),
        prefill_ms=_ms(end.t_gen_start_ns, first),
        decode_tokens_per_s=decode,
        total_ms=_ms(t_request_ns, end.t_end_ns),
    )
