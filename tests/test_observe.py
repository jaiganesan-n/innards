"""Non-blocking observer: bounded queue, background workers, fail-open behavior."""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import pytest

from innards.observer import BoundedEventQueue, EventKind, Observer, TurnEnd, TurnStart, TurnStats
from innards.schema import ModelInfo, RuntimeInfo, TurnRecord
from innards.sinks import CallbackSink, ConsoleSink, available_sinks, make_sink, make_sinks
from innards.sinks.console import format_bytes


def end(n: int = 3) -> TurnEnd:
    t = time.perf_counter_ns()
    return TurnEnd(
        generated_tokens=n,
        text="hi",
        finish_reason="length",
        cache_tokens=10,
        t_gen_start_ns=t,
        t_first_token_ns=t,
        t_last_token_ns=t,
        t_end_ns=t,
    )


def builder_recording_thread(seen: list[str]) -> Any:
    def build(start: TurnStart, end: TurnEnd, stats: TurnStats) -> TurnRecord:
        seen.append(threading.current_thread().name)
        record = TurnRecord(
            conversation="s",
            turn=1,
            timestamp="t",
            model=ModelInfo(id="m"),
            runtime=RuntimeInfo(backend="fake"),
        )
        record.tokens.generated = end.generated_tokens
        record.tokens.prompt = stats.tokens_seen  # smuggle the worker's count out for the test
        return record

    return build


def run_turn(obs: Observer, builder: Any, tokens: int = 3, key: tuple[str, int] = ("s", 1)) -> TurnRecord:
    future = obs.expect(key)
    obs.emit(EventKind.TURN_START, key, TurnStart(seed={}, t_request_ns=0, builder=builder))
    for _ in range(tokens):
        obs.emit(EventKind.TOKEN, key, 1)
    obs.emit(EventKind.TURN_END, key, end(tokens))
    return future.result(timeout=5)


# --------------------------------------------------------------------- queue


def test_queue_drops_and_counts_when_full() -> None:
    q: BoundedEventQueue[int] = BoundedEventQueue(maxsize=3)
    assert all(q.offer(i) for i in range(3))
    assert q.offer(99) is False
    assert q.offer(100) is False
    assert q.dropped == 2
    assert [q.get(0.1) for _ in range(3)] == [0, 1, 2]
    assert q.get(0.01) is None


def test_offer_never_blocks_on_full_queue() -> None:
    q: BoundedEventQueue[int] = BoundedEventQueue(maxsize=1)
    q.offer(0)
    start = time.perf_counter()
    for i in range(10_000):
        q.offer(i)
    assert time.perf_counter() - start < 1.0
    assert q.dropped == 10_000


def test_queue_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        BoundedEventQueue(0)


# --------------------------------------------------------------------- observer


def test_records_are_built_in_a_background_thread() -> None:
    seen: list[str] = []
    received: list[TurnRecord] = []
    obs = Observer([CallbackSink(received.append)]).start()
    record = run_turn(obs, builder_recording_thread(seen), tokens=5)
    obs.close()
    assert seen == ["innards-observer"]
    assert record.tokens.generated == 5
    assert record.tokens.prompt == 5  # worker saw 5 TOKEN events
    assert received == [record]
    assert obs.errors == 0
    assert obs.dropped == 0


def test_builder_error_is_fail_open() -> None:
    def broken(start: TurnStart, end: TurnEnd, stats: TurnStats) -> TurnRecord:
        raise RuntimeError("boom")

    obs = Observer().start()
    record = run_turn(obs, broken)
    obs.close()
    assert record.output.text == "hi"  # partial record keeps what is certain
    assert obs.errors == 1
    assert record.observer.errors == 1


def test_sink_error_is_fail_open_and_other_sinks_still_run() -> None:
    good: list[TurnRecord] = []

    def bad(record: TurnRecord) -> None:
        raise OSError("disk full")

    obs = Observer([CallbackSink(bad), CallbackSink(good.append)]).start()
    run_turn(obs, builder_recording_thread([]))
    obs.close()
    assert len(good) == 1
    assert obs.errors == 1


def test_emit_does_not_wait_for_a_stuck_worker() -> None:
    release = threading.Event()

    def slow(start: TurnStart, end: TurnEnd, stats: TurnStats) -> TurnRecord:
        release.wait(5)
        return builder_recording_thread([])(start, end, stats)

    obs = Observer(queue_size=8).start()
    key = ("s", 1)
    future = obs.expect(key)
    obs.emit(EventKind.TURN_START, key, TurnStart(seed={}, t_request_ns=0, builder=slow))
    obs.emit(EventKind.TURN_END, key, end())
    time.sleep(0.05)  # worker is now blocked inside the builder
    start = time.perf_counter()
    results = [obs.emit(EventKind.TOKEN, ("s", 2), 1) for _ in range(1000)]
    elapsed = time.perf_counter() - start
    release.set()
    record = future.result(timeout=5)
    obs.close()
    assert elapsed < 0.5
    assert results.count(False) >= 990  # queue of 8 filled, the rest dropped
    assert obs.dropped >= 990
    assert record.observer.dropped_events >= 990


def test_dropped_start_still_yields_partial_record() -> None:
    obs = Observer().start()
    key = ("s", 9)
    future = obs.expect(key)
    obs.emit(EventKind.TURN_END, key, end())
    record = future.result(timeout=5)
    obs.close()
    assert record.turn == 9
    assert record.output.text == "hi"
    assert obs.errors == 1


def test_memory_sampler_feeds_peak() -> None:
    values = iter([100, 300, 200, 150, 120, 110, 105])

    def sampler() -> int:
        return next(values, 50)

    captured: list[TurnStats] = []

    def build(start: TurnStart, end: TurnEnd, stats: TurnStats) -> TurnRecord:
        captured.append(stats)
        return builder_recording_thread([])(start, end, stats)

    obs = Observer(memory_sampler=sampler, sample_interval_s=0).start()
    run_turn(obs, build, tokens=4)
    obs.close()
    assert captured[0].peak_sampled_bytes == 300


def test_listener_and_sampler_errors_are_counted() -> None:
    def bad_sampler() -> int:
        raise RuntimeError("no /proc")

    obs = Observer(memory_sampler=bad_sampler).start()
    obs.add_listener(lambda record: 1 / 0)  # type: ignore[func-returns-value]
    record = run_turn(obs, builder_recording_thread([]), tokens=1)
    obs.close()
    assert record.tokens.generated == 1
    assert obs.errors >= 2


# --------------------------------------------------------------------- sinks


def test_make_sink_variants() -> None:
    assert "console" in available_sinks()
    assert isinstance(make_sink("console"), ConsoleSink)
    assert make_sink("console:json").fmt == "json"  # type: ignore[attr-defined]
    assert isinstance(make_sink(print), CallbackSink)
    sink = ConsoleSink()
    assert make_sink(sink) is sink
    assert len(make_sinks(["console", print])) == 2
    with pytest.raises(ValueError, match="Unknown sink"):
        make_sink("s3://bucket/")
    with pytest.raises(ValueError, match="Unknown console sink option"):
        make_sink("console:purple")


def test_console_sink_text_and_json() -> None:
    record = run_turn(Observer().start(), builder_recording_thread([]))
    record.kv.measured_bytes = 2048
    record.kv.predicted_bytes = 2048
    buf = io.StringIO()
    ConsoleSink(stream=buf).write(record)
    text = buf.getvalue()
    assert "[innards] turn 1" in text
    assert "KV measured 2.0 KiB vs predicted 2.0 KiB (+0.0%)" in text
    buf = io.StringIO()
    ConsoleSink(stream=buf, fmt="json").write(record)
    assert TurnRecord.from_json(buf.getvalue()) == record


def test_format_bytes() -> None:
    assert format_bytes(None) == "n/a"
    assert format_bytes(512) == "512 B"
    assert format_bytes(114_688) == "112.0 KiB"
    assert format_bytes(378_929_152) == "361.4 MiB"
