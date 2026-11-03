"""``innards.observe``: watch a model you already loaded. One turn per ``generate()`` call.

::

    import innards

    # ... your code loads tokenizer + model (original checkpoint) ...
    watch = innards.observe(model, tokenizer, model_id="Qwen/Qwen3.5-0.8B", run="deltanet-study")
    out = model.generate(**inputs, max_new_tokens=256)     # your code, unchanged
    watch.view()                                           # the web view, on this machine

What happens on the generation path, and nothing more:

* ``model.generate`` is wrapped (an instance attribute; :meth:`Watch.close` removes it). The wrapper
  takes a timestamp and the prompt's shape, calls the original ``generate`` with the same arguments,
  and afterwards copies the output ids to the CPU and the cache's tensor *metadata*.
* One forward pre-hook on the model counts steps (prefill, then one per decoded token), keeps a
  reference to the cache object and offers a small event to a bounded queue.
* Feature probes (hooks on the modules of detected features) reduce small tensors on the device and
  copy them to the CPU without blocking (see :mod:`innards.observer.probes`).

Everything else runs in background threads: the KV prediction and measurement, memory, timing, the
feature analyzers, writing the run folder and serving the view. Hooks are fail-open: an error is
counted and logged, and generation continues with the output it would have had.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import platform
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from innards import __version__
from innards._env import budget_bytes, device_name, environment, host_kind
from innards.collectors.memory import rss_sampler, split_memory, weights_bytes
from innards.collectors.timing import timing_from
from innards.features.base import Analyzer, Detection, TurnData
from innards.features.registry import analyzers as registered_analyzers
from innards.features.registry import detect, feature_status
from innards.kv.calculator import predict
from innards.kv.config import spec_from_config
from innards.kv.dtypes import normalize_dtype
from innards.lineage import explain
from innards.observer.cache import CacheSnapshot, measure_snapshot, snapshot_cache
from innards.observer.events import EventKind, TurnEnd, TurnKey, TurnStart, TurnStats
from innards.observer.observer import Observer
from innards.observer.probes import CapturedTensors, ProbeRecorder
from innards.schema import (
    ContextComposition,
    ConversationRecord,
    KVInfo,
    ModelInfo,
    ObserverInfo,
    OutputInfo,
    ProbeHealth,
    RuntimeInfo,
    TokenCounts,
    TurnRecord,
)
from innards.sinks import SinkSpec, make_sinks
from innards.store import RunStore, now_iso, safe_name

__all__ = ["CapturedTurn", "Watch", "observe"]

log = logging.getLogger("innards")

FeatureSpec = str | Sequence[str] | None
"""``"auto"`` (every detected feature), ``"light"`` (no hooks), a list of analyzer names, or ``None``."""


def _conversation_id(model_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{safe_name(model_id.rsplit('/', 1)[-1], 'model')[:40]}-{uuid.uuid4().hex[:4]}"


def _layers_snapshot(snapshot: CacheSnapshot | None) -> list[dict[str, Any]]:
    if snapshot is None:
        return []
    return [
        {
            "kind": layer.kind,
            "kv": [[list(shape), size] for shape, size in layer.kv],
            "state": [[list(shape), size] for shape, size in layer.state],
            "extra": [[group, list(meta[0]), meta[1]] for group, meta in layer.extra],
        }
        for layer in snapshot.layers
    ]


@dataclass
class CapturedTurn:
    """What ``observe`` captured on one turn: the record and each feature's raw captures (CPU tensors).

    Pass it to an analyzer explicitly: ``GatedDeltaNet(model="Qwen/Qwen3.5-0.8B", args=watch.last_turn())``.
    """

    record: TurnRecord
    data: dict[str, TurnData]
    """Feature name → the turn's captures for that feature."""
    _base: TurnData | None = field(default=None, repr=False)

    def turn_data(self, feature: str, detection: Detection) -> TurnData:
        """Captures for ``feature``; for a light analyzer that was not run, the shared cache layout and record."""
        if feature in self.data:
            return self.data[feature]
        if self._base is None:
            raise ValueError("This turn carries no captures")
        return TurnData(
            feature=feature,
            detection=detection,
            captures={},
            prefill_tokens=self._base.prefill_tokens,
            token_ids=self._base.token_ids,
            record=self._base.record,
            config=self._base.config,
            cache_layers=self._base.cache_layers,
            decode=self._base.decode,
        )

    @property
    def features(self) -> dict[str, Any]:
        return self.record.features


@dataclass
class _Live:
    """The turn running on the generation thread (owned by that thread until the end event)."""

    key: TurnKey
    t_request: int
    prompt: int
    batch: int
    message: str | None
    set_step: dict[str, Any] | None
    steps: int = 0
    first_len: int | None = None
    t_gen_start: int | None = None
    t_first: int | None = None
    cache: Any = None
    hot_ns: int = 0


class _StoreSink:
    """Writes each finished record to the run folder (in the observer's sink thread)."""

    def __init__(self, watch: Watch) -> None:
        self._watch = watch

    def write(self, record: TurnRecord) -> None:
        self._watch._persist(record)

    def close(self) -> None:
        return None


class Watch:
    """Hooks on one loaded model, and what they have measured. Create it with :func:`observe`."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        *,
        model_id: str | None = None,
        run: str = "default",
        root: str | None = None,
        features: FeatureSpec = "auto",
        sample_every: int = 1,
        layers: Iterable[int] | None = None,
        budget_gb: float | None = None,
        label: str | None = None,
        conversation_set: str | None = None,
        store: bool = True,
        sinks: Iterable[SinkSpec] | None = None,
        queue_size: int = 4096,
        record_timeout: float = 120.0,
        settings: dict[str, Any] | None = None,
    ) -> None:
        if getattr(model, "_innards_watch", None) is not None:
            raise RuntimeError("This model is already observed; call watch.close() on the other Watch first")
        if not callable(getattr(model, "generate", None)):
            raise TypeError("observe() needs a Hugging Face model with generate() (e.g. from AutoModelForCausalLM)")
        self.model = model
        self.tokenizer = tokenizer
        config = model.config.to_dict() if hasattr(model.config, "to_dict") else dict(model.config)
        self.config: dict[str, Any] = config
        self.model_id = str(
            model_id
            or getattr(model.config, "_name_or_path", None)
            or getattr(model, "name_or_path", None)
            or type(model).__name__
        )
        self.spec = spec_from_config(config, model_id=self.model_id)
        param = next(model.parameters())
        self.device = param.device
        self.device_type = str(param.device.type)
        compute = getattr(model, "dtype", param.dtype)
        try:
            dtype = normalize_dtype(compute)
        except ValueError:
            dtype = "bfloat16"
        self.dtype = "bfloat16" if dtype == "float8" else dtype
        self.kv_dtype = self.dtype
        self.weights_bytes = weights_bytes(model)
        self.budget_bytes = budget_bytes(budget_gb)
        self.record_timeout = record_timeout
        self.run = safe_name(run or "default")
        self.store = RunStore(root) if store else None
        self._settings = dict(settings or {})
        self._environment = environment()
        self.records: list[TurnRecord] = []
        self._lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._live: _Live | None = None
        self._last_capture: CapturedTurn | None = None
        self._next: dict[str, Any] = {}
        self._submitted = 0
        self._closed = False

        # ------------------------------------------------------------ features and probes
        selected = self._select(features)
        self._recorder = ProbeRecorder(self.device_type, sample_every)
        layer_set = set(layers) if layers is not None else None
        self.analyzers: list[tuple[Analyzer, Detection]] = []
        for analyzer, detection in selected:
            hooks = analyzer.attach(model, detection, self._recorder, layer_set)
            if hooks or analyzer.light:
                self.analyzers.append((analyzer, detection))
            else:
                log.warning("innards: %s is in the config but no matching modules were found", analyzer.title)
        self.feature_status = feature_status(config, active=[a.name for a, _ in self.analyzers])
        self._model_info = self._build_model_info()
        self._runtime = RuntimeInfo(
            backend="observe",
            host=host_kind(),
            device=device_name(self.device),
            python=platform.python_version(),
            engine_version=(
                f"transformers {self._environment.get('transformers')}; torch {self._environment.get('torch')}"
            ),
            innards_version=__version__,
        )

        # ------------------------------------------------------------ observer, then the two generation-path hooks
        sink_list = ([_StoreSink(self)] if self.store is not None else []) + make_sinks(sinks)
        self.observer = Observer(
            sink_list, queue_size=queue_size, memory_sampler=self._sampler(), name=f"innards-{self.run}"
        ).start()
        self.observer.add_listener(self._on_record)
        self._had_generate = "generate" in vars(model)
        self._original_generate: Callable[..., Any] = model.generate
        self._step_handle = model.register_forward_pre_hook(self._on_forward, with_kwargs=True)
        model.generate = self._generate
        model._innards_watch = self
        self._start_conversation(label, conversation_set)

    # ------------------------------------------------------------------ setup helpers

    def _select(self, features: FeatureSpec) -> list[tuple[Analyzer, Detection]]:
        found = detect(self.config)
        if features == "auto":
            return found
        if features == "light":
            return [(a, d) for a, d in found if a.light]
        if features is None or features == "none":
            names = {"kv"}
        elif isinstance(features, str):
            names = {"kv", features}
        else:
            names = {"kv", *features}
        known = {a.name for a in registered_analyzers()}
        unknown = sorted(names - known)
        if unknown:
            raise ValueError(f"Unknown feature analyzers {unknown}; known: {', '.join(sorted(known))}")
        chosen = [(a, d) for a, d in found if a.name in names]
        missing = sorted(names - {a.name for a, _ in chosen})
        if missing:
            log.warning("innards: this model does not have %s; nothing to analyze there", ", ".join(missing))
        return chosen

    def _build_model_info(self) -> ModelInfo:
        spec = self.spec
        quant = self.config.get("quantization_config")
        method = quant.get("quant_method") if isinstance(quant, dict) else None
        try:
            lineage = explain(spec).summary()
        except KeyError:
            lineage = None
        return ModelInfo(
            id=self.model_id,
            architecture=spec.architecture,
            attention=spec.attention,
            lineage=lineage,
            layers=spec.num_layers,
            kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            sliding_window=spec.sliding_window,
            dtype=self.dtype,
            kv_dtype=self.kv_dtype,
            quantization=str(getattr(method, "value", method)) if method else None,
            model_type=spec.model_type,
            features=[a.name for a, _ in self.analyzers],
            parameters=sum(int(p.numel()) for p in self.model.parameters()),
        )

    def _sampler(self) -> Callable[[], int | None] | None:
        import torch

        if self.device_type == "cuda":
            device = self.device
            return lambda: int(torch.cuda.memory_allocated(device))
        if self.device_type == "mps":
            return lambda: int(torch.mps.current_allocated_memory())
        return rss_sampler()

    def _start_conversation(self, label: str | None, conversation_set: str | None) -> None:
        self.conversation = _conversation_id(self.model_id)
        self._turn = 0
        self._last_measured: int | None = None
        self._analyzer_state: dict[str, dict[str, Any]] = {a.name: {} for a, _ in self.analyzers}
        self._conversation = ConversationRecord(
            id=self.conversation,
            model=copy.copy(self._model_info),
            run=self.run,
            started_at=now_iso(),
            status="active",
            set=conversation_set,
            label=label or self.model_id.rsplit("/", 1)[-1],
            runtime=self._runtime,
            environment=self._environment,
            settings=dict(self._settings),
            feature_status={k: v["status"] for k, v in self.feature_status.items()},
            config=self.config,
        )
        self._stored = False

    # ------------------------------------------------------------------ public API

    @property
    def turns(self) -> int:
        """Turns in the current conversation."""
        return self._turn

    @property
    def dropped_events(self) -> int:
        return self.observer.dropped

    @property
    def errors(self) -> int:
        return self.observer.errors

    @property
    def generating(self) -> bool:
        return self._live is not None

    def next_turn(
        self, message: str | None = None, *, set_name: str | None = None, step: int | None = None, of: int | None = None
    ) -> Watch:
        """Label the next ``generate()`` call: the user's message and, for a conversation set, its step."""
        with self._lock:
            self._next = {
                "message": message,
                "set": {"name": set_name, "step": step, "of": of} if set_name else None,
            }
        return self

    def new_conversation(
        self, label: str | None = None, conversation_set: str | None = None, settings: dict[str, Any] | None = None
    ) -> str:
        """Finish the current conversation and start another (same model, same run). Returns its id."""
        with self._turn_lock:
            self.wait()
            self._finish_conversation("completed")
            if settings is not None:
                self._settings = dict(settings)
            self._start_conversation(label, conversation_set)
        return self.conversation

    def wait(self, timeout: float | None = None) -> None:
        """Wait until every finished turn has been analyzed and written to the run folder."""
        deadline = time.monotonic() + (timeout if timeout is not None else self.record_timeout)
        while time.monotonic() < deadline:
            observer = self.observer
            if observer.processed >= self._submitted and observer.sunk + observer.sink_dropped >= observer.processed:
                return
            time.sleep(0.005)

    def last_turn(self, timeout: float | None = None) -> CapturedTurn:
        """The most recent turn as :class:`CapturedTurn` (waits for its analysis)."""
        self.wait(timeout)
        if self._last_capture is None:
            raise RuntimeError("No turn observed yet: call model.generate() first")
        return self._last_capture

    def status(self) -> dict[str, Any]:
        """Live state for the view: model, current turn, light-tier values, features and probe health."""
        live = self._live
        last = self.records[-1] if self.records else None
        allocated = None
        sampler = self.observer._memory_sampler
        if sampler is not None:
            try:
                allocated = sampler()
            except Exception:
                allocated = None
        return {
            "model": self._model_info_dict(),
            "run": self.run,
            "conversation": self.conversation,
            "turns": self._turn,
            "live": {
                "generating": live is not None,
                "turn": live.key[1] if live else self._turn,
                "generated": max((live.steps if live else 0) - 1, 0),
                "prompt_tokens": live.prompt if live else None,
                "elapsed_ms": round((time.perf_counter_ns() - live.t_request) / 1e6, 1) if live else None,
            },
            "last": last.to_dict() if last else None,
            "features": self.feature_status,
            "health": {
                "overhead_pct": last.observer.overhead_pct if last else None,
                "dropped": self.observer.dropped,
                "errors": self.observer.errors,
                "probes": [asdict(p) for p in last.observer.probes] if last else [],
            },
            "memory": {
                "allocated_bytes": allocated,
                "budget_bytes": self.budget_bytes,
                "weights_bytes": self.weights_bytes,
                "device": self._runtime.device,
            },
            "store": self.store.url if self.store else None,
        }

    def _model_info_dict(self) -> dict[str, Any]:
        return asdict(self._model_info)

    def view(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        open_browser: bool = False,
        runner: Any = None,
        block: bool = False,
    ) -> Any:
        """Serve the web view from this process (background thread). Returns the server; ``.url``, ``.stop()``.

        Needs ``pip install "innards[view]"``. Pass ``runner`` to let the page load models and send messages.
        """
        from innards.view import serve

        root = str(self.store.root) if self.store else None
        return serve(root=root, watch=self, runner=runner, host=host, port=port, open_browser=open_browser, block=block)

    def sync(self, url: str, client: Any = None) -> Any:
        """Upload this run's folder to S3 (``s3://bucket/prefix``). Needs the ``aws`` extra."""
        if self.store is None:
            raise RuntimeError("This watch does not store results (store=False)")
        from innards.sync import sync_up

        self.wait()
        return sync_up(self.store, self.run, url, client=client)

    def close(self) -> None:
        """Remove every hook and the ``generate`` wrapper, finish analysis and close the conversation."""
        if self._closed:
            return
        with self._turn_lock:
            self._closed = True
            self._recorder.detach()
            self._step_handle.remove()
            if self._had_generate:
                self.model.generate = self._original_generate
            else:
                with contextlib.suppress(AttributeError):
                    del self.model.generate
            with contextlib.suppress(AttributeError):
                del self.model._innards_watch
            self.wait()
            self.observer.close()
            self._finish_conversation("completed")

    def __enter__(self) -> Watch:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ generation path

    def _on_forward(self, module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        live = self._live
        if live is None:
            return None
        now = time.perf_counter_ns()
        live.steps += 1
        if live.steps == 1:
            live.t_gen_start = now
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            if ids is None:
                ids = kwargs.get("inputs_embeds")
            if ids is not None and hasattr(ids, "shape") and len(ids.shape) >= 2:
                live.first_len = int(ids.shape[1])
        elif live.steps == 2:
            live.t_first = now
        cache = kwargs.get("past_key_values")
        if cache is not None:
            live.cache = cache
        self._recorder.on_step()
        if live.steps > 1:
            self.observer.emit(EventKind.TOKEN, live.key, 1)
        live.hot_ns += time.perf_counter_ns() - now
        return None

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        if self._closed or not self._turn_lock.acquire(blocking=False):  # nested or concurrent: pass through
            return self._original_generate(*args, **kwargs)
        try:
            return self._observed_generate(args, kwargs)
        finally:
            self._turn_lock.release()

    def _observed_generate(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        t_request = time.perf_counter_ns()
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = kwargs.get("inputs", args[0] if args else None)
        embeds = kwargs.get("inputs_embeds")
        shape = getattr(ids if ids is not None else embeds, "shape", None)
        prompt = int(shape[1]) if shape is not None and len(shape) >= 2 else 0
        batch = int(shape[0]) if shape is not None and len(shape) >= 1 else 1
        with self._lock:
            labels, self._next = self._next, {}
        self._turn += 1
        key: TurnKey = (self.conversation, self._turn)
        live = _Live(key, t_request, prompt, batch, labels.get("message"), labels.get("set"))
        seed = {
            "conversation": self.conversation,
            "turn": self._turn,
            "timestamp": now_iso(),
            "prompt": prompt,
            "batch": batch,
            "message": live.message,
            "set": live.set_step,
            "has_ids": ids is not None,
        }
        self._submitted += 1
        self.observer.emit_control(EventKind.TURN_START, key, TurnStart(seed, t_request, self._build))
        self._recorder.begin_turn()
        self._live = live
        live.hot_ns += time.perf_counter_ns() - t_request
        try:
            out = self._original_generate(*args, **kwargs)
        except BaseException as exc:
            self._live = None
            self._recorder.abort_turn()
            self.observer.emit_control(
                EventKind.TURN_END,
                key,
                TurnEnd(
                    0,
                    "",
                    "error",
                    None,
                    live.t_gen_start,
                    live.t_first,
                    None,
                    time.perf_counter_ns(),
                    error=repr(exc),
                    extra={"hot_ns": live.hot_ns},
                ),
            )
            raise
        t_end = time.perf_counter_ns()
        self._live = None
        end = self._collect(live, out, ids is not None, kwargs, t_end)
        self.observer.emit_control(EventKind.TURN_END, key, end)
        return out

    def _collect(self, live: _Live, out: Any, has_ids: bool, kwargs: dict[str, Any], t_end: int) -> TurnEnd:
        """Right after generate returns: CPU copy of the ids, cache metadata, probe hand-over."""
        sequences = out if hasattr(out, "shape") else getattr(out, "sequences", None)
        cached = max(live.prompt - (live.first_len or live.prompt), 0) if has_ids else 0
        row: list[int] = []
        if sequences is not None and len(sequences.shape) == 2:
            start = cached if has_ids else 0
            row = [int(t) for t in sequences[0, start:].tolist()]
        new_prompt = row[: live.prompt - cached] if has_ids else []
        generated = row[live.prompt - cached :] if has_ids else row
        cache = live.cache if live.cache is not None else getattr(out, "past_key_values", None)
        try:
            snapshot = snapshot_cache(cache)
        except Exception:
            log.warning("innards: could not read this cache layout; KV will be null", exc_info=True)
            snapshot = None
        captured = self._recorder.end_turn(cache)
        counters: dict[str, int] = {}
        if self.device_type == "cuda":
            import torch

            counters = {
                "allocated": int(torch.cuda.memory_allocated(self.device)),
                "peak": int(torch.cuda.max_memory_allocated(self.device)),
            }
        n = len(generated)
        pad = getattr(getattr(self.model, "generation_config", None), "pad_token_id", None)
        if live.batch > 1 and pad is not None:
            while generated and generated[-1] == pad and len(generated) > 1:
                generated.pop()
            n = len(generated)
        max_new = kwargs.get("max_new_tokens") or getattr(
            getattr(self.model, "generation_config", None), "max_new_tokens", None
        )
        finish = "length" if max_new and n >= int(max_new) else "stop"
        cache_tokens = live.prompt + n - 1 if cache is not None and n else (live.prompt if cache is not None else None)
        return TurnEnd(
            generated_tokens=n,
            text="",
            finish_reason=finish,
            cache_tokens=cache_tokens,
            t_gen_start_ns=live.t_gen_start,
            t_first_token_ns=live.t_first if n > 1 else t_end,
            t_last_token_ns=t_end,
            t_end_ns=t_end,
            cache_snapshot=snapshot,
            counters=counters,
            extra={
                "hot_ns": live.hot_ns + (time.perf_counter_ns() - t_end),
                "captured": captured,
                "cached": cached,
                "new_prompt": new_prompt,
                "generated": generated,
            },
        )

    # ------------------------------------------------------------------ worker side (observer thread)

    def _decode(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            return ""
        return str(self.tokenizer.decode(ids, skip_special_tokens=True))

    def _message(self, seed: dict[str, Any], new_prompt: list[int]) -> str | None:
        if seed.get("message") is not None:
            return str(seed["message"])
        if self.tokenizer is None or not new_prompt:
            return None
        text = self._decode(new_prompt)
        lines = [ln for ln in text.splitlines() if ln.strip().lower() not in {"user", "assistant", "system", "model"}]
        return "\n".join(lines).strip() or None

    def _build(self, start: TurnStart, end: TurnEnd, stats: TurnStats) -> TurnRecord:
        seed = start.seed
        extra = end.extra
        captured: CapturedTensors | None = extra.get("captured")
        record = TurnRecord(
            conversation=seed["conversation"],
            turn=seed["turn"],
            timestamp=seed["timestamp"],
            model=copy.copy(self._model_info),
            runtime=copy.copy(self._runtime),
            run=self.run,
            message=self._message(seed, extra.get("new_prompt") or []),
            set=seed.get("set"),
        )
        prompt, cached = int(seed["prompt"]), int(extra.get("cached") or 0)
        generated = end.generated_tokens
        record.tokens = TokenCounts(
            prompt=prompt,
            cached=cached,
            recomputed=prompt - cached,
            generated=generated,
            context_after=prompt + generated,
            by_modality={"text": prompt + generated},
            batch_size=int(seed.get("batch") or 1),
        )
        record.context = ContextComposition(history=cached, new_message=prompt - cached)
        kv = KVInfo(prefix_reused=cached > 0, cache_tokens=end.cache_tokens)
        snapshot = end.cache_snapshot
        reading = measure_snapshot(snapshot) if snapshot is not None else None
        kv_dtype = self.kv_dtype
        if snapshot is not None:
            sizes = [meta[1] for layer in snapshot.layers for meta in layer.kv]
            if sizes:
                kv_dtype = {
                    4: "float32",
                    2: self.kv_dtype if self.kv_dtype in {"float16", "bfloat16"} else "bfloat16",
                    1: "float8",
                }.get(sizes[0], self.kv_dtype)
        if end.cache_tokens is not None:
            memory = max(self.budget_bytes - self.weights_bytes, 0)
            prediction = predict(
                self.spec, end.cache_tokens, kv_dtype, memory_bytes=memory, batch_size=int(seed.get("batch") or 1)
            )
            kv.predicted_bytes = prediction.total_bytes
            kv.predicted_by_layer_type = dict(prediction.by_layer_type)
            kv.predicted_per_layer_bytes = list(prediction.per_layer_bytes)
            kv.capacity_tokens = prediction.capacity_tokens
        if reading is not None:
            kv.measured_bytes = reading.total_bytes
            kv.measured_source = reading.source
            kv.by_layer_type = dict(reading.by_layer_type)
            kv.per_layer_bytes = list(reading.per_layer_bytes)
            kv.growth_bytes = reading.total_bytes - (self._last_measured or 0)
            self._last_measured = reading.total_bytes
        if kv.capacity_tokens is not None:
            per_turn = (prompt + generated) / max(int(seed["turn"]), 1)
            remaining = kv.capacity_tokens - (prompt + generated)
            kv.turns_to_limit = max(int(remaining // per_turn), 0) if per_turn > 0 else None
        record.kv = kv
        if self.device_type == "cuda":
            record.memory = split_memory(
                weights_bytes=self.weights_bytes,
                kv_bytes=kv.measured_bytes,
                peak_bytes=max(end.counters.get("peak", 0), stats.peak_sampled_bytes or 0) or None,
                allocated_bytes=end.counters.get("allocated"),
                budget_bytes=self.budget_bytes,
                source="torch_cuda",
                allocator_only=True,
            )
        else:
            record.memory = split_memory(
                weights_bytes=self.weights_bytes,
                kv_bytes=kv.measured_bytes,
                peak_bytes=stats.peak_sampled_bytes,
                budget_bytes=self.budget_bytes,
                source="torch_mps" if self.device_type == "mps" else "psutil_rss",
                allocator_only=self.device_type == "mps",
            )
        record.timing = timing_from(start.t_request_ns, end)
        record.output = OutputInfo(text=self._decode(extra.get("generated") or []), finish_reason=end.finish_reason)
        if end.error:
            record.output = OutputInfo(text=None, finish_reason="error")
        total_ns = max(end.t_end_ns - start.t_request_ns, 1)
        hot = int(extra.get("hot_ns") or 0)
        probes: list[ProbeHealth] = []
        if captured is not None:
            for feature, hooks in captured.hooks.items():
                spent = captured.overhead_ns.get(feature, 0)
                hot += spent
                probes.append(
                    ProbeHealth(
                        feature=feature,
                        hooks=hooks,
                        overhead_ms=round(spent / 1e6, 3),
                        overhead_pct=round(spent / total_ns * 100, 3),
                        sampled_tokens=captured.sampled_steps,
                        errors=captured.errors.get(feature, 0),
                    )
                )
        record.observer = ObserverInfo(
            overhead_ms=round(hot / 1e6, 3), overhead_pct=round(hot / total_ns * 100, 3), probes=probes
        )
        if not end.error:
            record.features, data = self._analyze(record, snapshot, captured, extra)
            self._last_capture = CapturedTurn(record, data, data.get("__base__"))
            data.pop("__base__", None)
        return record

    def _analyze(
        self,
        record: TurnRecord,
        snapshot: CacheSnapshot | None,
        captured: CapturedTensors | None,
        extra: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if captured is not None:
            captured.wait()  # only this worker waits for the non-blocking copies
        base_record = record.to_dict()
        cache_layers = _layers_snapshot(snapshot)
        new_prompt = list(extra.get("new_prompt") or [])
        token_ids = new_prompt + list(extra.get("generated") or [])
        decode = self.tokenizer.decode if self.tokenizer is not None else None
        payloads: dict[str, dict[str, Any]] = {}
        data: dict[str, Any] = {}
        base = TurnData(
            feature="",
            detection=Detection(""),
            captures={},
            prefill_tokens=len(new_prompt),
            token_ids=token_ids,
            record=base_record,
            config=self.config,
            cache_layers=cache_layers,
            decode=decode,
        )
        data["__base__"] = base
        for analyzer, detection in self.analyzers:
            turn = TurnData(
                feature=analyzer.name,
                detection=detection,
                captures=captured.by_feature(analyzer.name) if captured is not None else {},
                prefill_tokens=len(new_prompt),
                token_ids=token_ids,
                record=base_record,
                config=self.config,
                cache_layers=cache_layers,
                state=self._analyzer_state.setdefault(analyzer.name, {}),
                decode=decode,
                extras={analyzer.name: captured.extras.get(analyzer.name)} if captured is not None else {},
            )
            data[analyzer.name] = turn
            try:
                payload = analyzer.analyze(turn)
            except Exception:
                self.observer._count_error("analyzer %s", analyzer.name)
                continue
            if payload is not None:
                payloads[analyzer.name] = payload
        return payloads, data

    def _on_record(self, record: TurnRecord) -> None:
        self.records.append(record)
        if len(self.records) > 1000:  # the run folder keeps everything; memory keeps the recent turns
            del self.records[:-1000]

    def _persist(self, record: TurnRecord) -> None:
        if self.store is None or record.conversation != self._conversation.id:
            if self.store is not None:
                self.store.append_turn(self.run, record.conversation, record)
            return
        if not self._stored:
            self.store.ensure_run(self.run, self._environment)
            self._stored = True
        self.store.append_turn(self.run, record.conversation, record)
        self._conversation.turns = max(self._conversation.turns, record.turn)
        self.store.write_conversation(self.run, self._conversation)

    def _finish_conversation(self, status: str) -> None:
        if self.store is None or not self._stored:
            return
        self._conversation.status = status
        try:
            self.store.write_conversation(self.run, self._conversation)
        except Exception:
            log.exception("innards: could not finalize conversation %s", self._conversation.id)


def observe(
    model: Any,
    tokenizer: Any = None,
    *,
    model_id: str | None = None,
    run: str = "default",
    root: str | None = None,
    features: FeatureSpec = "auto",
    sample_every: int = 1,
    layers: Iterable[int] | None = None,
    budget_gb: float | None = None,
    label: str | None = None,
    conversation_set: str | None = None,
    store: bool = True,
    sinks: Iterable[SinkSpec] | None = None,
    queue_size: int = 4096,
    settings: dict[str, Any] | None = None,
) -> Watch:
    """Attach read-only hooks to a model you already loaded. Each ``generate()`` call becomes one turn.

    Args:
        model: A loaded Hugging Face model (anything with ``generate`` and ``config``), original checkpoint.
        tokenizer: Its tokenizer, used to show text next to per-token measurements (optional).
        model_id: The Hub id or path it came from (default: ``config._name_or_path``).
        run: Run name. Results go to ``<root>/<run>/<conversation>/``.
        root: Folder of runs (default ``$INNARDS_RUNS`` or ``./innards_runs``).
        features: ``"auto"`` runs every analyzer whose feature the config has; ``"light"`` attaches no
            probes (KV, context, memory and timing only); a list names analyzers (``["gated_deltanet"]``).
        sample_every: Record feature probes on the prefill and every Nth decoded token.
        layers: Attach feature probes to these layer indices only.
        budget_gb: Memory budget for capacity and fit (default ``$INNARDS_GPU_BUDGET_GB`` or 16).
        label / conversation_set / settings: Stored with the conversation (the runner sets them).
        store: ``False`` keeps records in memory only (``watch.records``).
        sinks: Extra destinations for each record, e.g. ``["console"]``.

    Returns:
        A :class:`Watch`. ``watch.view()`` serves the web view; ``watch.close()`` removes every hook.
    """
    return Watch(
        model,
        tokenizer,
        model_id=model_id,
        run=run,
        root=root,
        features=features,
        sample_every=sample_every,
        layers=layers,
        budget_gb=budget_gb,
        label=label,
        conversation_set=conversation_set,
        store=store,
        sinks=sinks,
        queue_size=queue_size,
        settings=settings,
    )
