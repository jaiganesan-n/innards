"""The probe recorder: what feature hooks call on the generation path.

Every hook registered through :meth:`ProbeRecorder.hook` is wrapped so that it

* runs only while a turn is being observed and the current step is sampled,
* is timed (host time, per feature) so each probe reports its own overhead,
* is fail-open: an exception is counted and logged once, never raised into the model,
* never changes the module's output (the wrapper returns ``None``).

Tensors handed to :meth:`ProbeRecorder.put` are copied to the CPU without blocking the generation
thread. On CUDA the copy goes to pinned host memory with ``non_blocking=True`` on the current
stream; a CUDA event recorded at turn end tells the worker when the copies are complete, so only the
worker ever waits. On CPU the tensor is cloned. On Apple MPS, where asynchronous device-to-host
copies are not reliable, tensors stay on the device during the turn and are copied once when
``generate`` returns (that copy is counted in the probe's overhead).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CapturedTensors", "ProbeRecorder"]

log = logging.getLogger("innards.observe")


@dataclass
class CapturedTensors:
    """Everything the probes recorded during one turn. Handed to the observer worker."""

    items: list[tuple[str, str, int, int, Any]]
    """``(feature, name, layer, step, tensor)``."""
    extras: dict[str, Any]
    overhead_ns: dict[str, int]
    errors: dict[str, int]
    hooks: dict[str, int]
    sampled_steps: int
    event: Any = None
    """CUDA event recorded after the last copy was queued (``None`` on CPU and MPS)."""
    _ready: bool = field(default=False, repr=False)

    def wait(self) -> None:
        """Block the calling (worker) thread until every non-blocking copy has landed."""
        if not self._ready and self.event is not None:
            self.event.synchronize()
        self._ready = True

    def by_feature(self, feature: str) -> dict[str, dict[int, list[tuple[int, Any]]]]:
        out: dict[str, dict[int, list[tuple[int, Any]]]] = {}
        for feat, name, layer, step, tensor in self.items:
            if feat == feature:
                out.setdefault(name, {}).setdefault(layer, []).append((step, tensor))
        return out


class ProbeRecorder:
    """Owns the feature hooks of one watched model and the tensors they capture in a turn."""

    def __init__(self, device_type: str, sample_every: int = 1) -> None:
        if sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        self.device_type = device_type
        self.sample_every = sample_every
        self.hooks: dict[str, int] = {}
        self._handles: list[Any] = []
        self._end: list[tuple[str, Callable[[Any], Any]]] = []
        self._logged: set[str] = set()
        self._reset()
        self.active = False

    def _reset(self) -> None:
        self.items: list[tuple[str, str, int, int, Any]] = []
        self.overhead_ns: dict[str, int] = {}
        self.errors: dict[str, int] = {}
        self.step = -1
        self.recording = False
        self.sampled_steps = 0

    # ------------------------------------------------------------------ turn lifecycle (generation thread)

    def begin_turn(self) -> None:
        self._reset()
        self.active = True

    def on_step(self) -> None:
        """Called once per forward pass: step 0 is the prefill, then one step per decoded token."""
        self.step += 1
        self.recording = self.step == 0 or self.step % self.sample_every == 0
        if self.recording:
            self.sampled_steps += 1

    def end_turn(self, cache: Any) -> CapturedTensors:
        """Run turn-end callbacks and hand the captures over. Called right after ``generate`` returns."""
        extras: dict[str, Any] = {}
        for feature, fn in self._end:
            start = time.perf_counter_ns()
            try:
                extras[feature] = fn(cache)
            except Exception:
                self._error(feature, "turn-end callback")
            finally:
                self._charge(feature, start)
        event = None
        if self.device_type == "cuda" and (self.items or extras):
            import torch

            event = torch.cuda.Event()  # type: ignore[no-untyped-call]
            event.record()
        elif self.device_type == "mps" and self.items:
            start = time.perf_counter_ns()
            try:
                self.items = [(f, n, layer, s, t.cpu()) for f, n, layer, s, t in self.items]
            finally:
                share = (time.perf_counter_ns() - start) // max(len({i[0] for i in self.items}), 1)
                for feature in {i[0] for i in self.items}:
                    self.overhead_ns[feature] = self.overhead_ns.get(feature, 0) + share
        self.active = False
        captured = CapturedTensors(
            items=self.items,
            extras=extras,
            overhead_ns=dict(self.overhead_ns),
            errors=dict(self.errors),
            hooks=dict(self.hooks),
            sampled_steps=self.sampled_steps,
            event=event,
        )
        self._reset()
        return captured

    def abort_turn(self) -> None:
        self.active = False
        self._reset()

    # ------------------------------------------------------------------ Recorder protocol

    def hook(
        self,
        feature: str,
        module: Any,
        fn: Callable[..., None],
        *,
        pre: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> None:
        recorder = self

        def wrapped(*args: Any) -> None:
            if not recorder.active or not recorder.recording:
                return None
            start = time.perf_counter_ns()
            try:
                fn(*args)
            except Exception:
                recorder._error(feature, type(args[0]).__name__)
            finally:
                recorder._charge(feature, start)
            return None  # never replace the module's inputs or output

        if pre:
            self._handles.append(module.register_forward_pre_hook(wrapped, with_kwargs=with_kwargs))
        else:
            self._handles.append(module.register_forward_hook(wrapped, always_call=always_call))
        self.hooks[feature] = self.hooks.get(feature, 0) + 1

    def put(self, feature: str, name: str, layer: int, tensor: Any) -> None:
        self.items.append((feature, name, layer, self.step, self.to_cpu(tensor)))

    def to_cpu(self, tensor: Any) -> Any:
        t = tensor.detach()
        if self.device_type == "cuda" and t.is_cuda:
            import torch

            buffer = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            buffer.copy_(t, non_blocking=True)
            return buffer
        if self.device_type == "mps":
            return t.clone()  # copied to the CPU once, when generate returns
        return t.clone()

    def at_turn_end(self, feature: str, fn: Callable[[Any], Any]) -> None:
        self._end.append((feature, fn))

    # ------------------------------------------------------------------ bookkeeping

    def _charge(self, feature: str, start: int) -> None:
        self.overhead_ns[feature] = self.overhead_ns.get(feature, 0) + time.perf_counter_ns() - start

    def _error(self, feature: str, where: str) -> None:
        self.errors[feature] = self.errors.get(feature, 0) + 1
        if feature not in self._logged:  # log the first failure of each probe, count the rest
            self._logged.add(feature)
            log.warning("innards: %s probe failed in %s; generation continues", feature, where, exc_info=True)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._end.clear()
        self.active = False
