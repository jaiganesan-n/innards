"""``innards.experts.trace``: which experts every token went to, by topic, while your code runs the model.

::

    import innards

    trace = innards.experts.trace(model, tokenizer, run="topic-experts")
    with trace.topic("math"):
        model.generate(**inputs, max_new_tokens=64)     # inference: the prefill and every decoded token
    with trace.topic(batch["topic"]):                  # training: one label per sequence in the batch
        loss = model(**batch).loss
    loss.backward()
    experts = trace.map()                              # topic x layer x expert, from what was measured

On the generation or training path, and nothing more:

* One forward pre-hook on the model reads the shape of ``input_ids``, the attention mask (padding is
  never counted) and the topic labels; one forward hook closes the forward. Forwards outside a topic,
  and recomputation during backward (gradient checkpointing), are not recorded.
* Per MoE layer, the hooks of :mod:`innards.experts.routing` hand over the chosen experts and their
  weights; the counts are added on the device with ``index_add_`` (no host synchronization).
* With ``tokens=True`` the chosen experts of every token stay on the device until :meth:`flush`.

:meth:`flush` (also run every ``flush_every`` seconds between forwards, and by :meth:`close`) is the
only place that waits for the device: it adds the counts to the totals and writes
``<run>/experts/<name>/summary.json`` and ``tokens/shard-*.safetensors``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from innards import __version__
from innards._env import environment
from innards.experts.routing import MoELayer, attach, find_moe_layers, routing_facts
from innards.store import RunStore, now_iso, safe_name

__all__ = ["FORMAT", "ExpertTrace", "trace"]

log = logging.getLogger("innards.experts")

FORMAT = "innards.experts/1.0"
TOKENS_FORMAT = "innards.experts.tokens/1.0"

Label = str | Sequence[str] | None


@dataclass
class _Forward:
    """One recorded forward pass: per token row (``batch x seq``, row-major), on the device."""

    n: int
    valid: Any
    topic: Any
    token: Any
    sequence: Any
    position: Any
    keep: bool
    routes: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    """Layer slot → (experts[n, k] int16, weights[n, k] float32), when tokens are kept."""


class _Scope:
    """``trace.topic(...)``: sets the label at once; as a context manager it restores the previous one."""

    def __init__(self, trace: ExpertTrace, previous: tuple[Label, bool]) -> None:
        self._trace = trace
        self._previous = previous

    def __enter__(self) -> ExpertTrace:
        return self._trace

    def __exit__(self, *exc: object) -> None:
        self._trace._label, self._trace._keep = self._previous
        self._trace._rows = {}


def _past_tokens(kwargs: dict[str, Any]) -> int:
    """Tokens already in the cache passed to this forward, from cache metadata (never a device read)."""
    cache = kwargs.get("past_key_values")
    length = getattr(cache, "get_seq_length", None)
    if callable(length):
        try:
            value = length()
        except Exception:
            return 0
        return value if isinstance(value, int) else 0
    return 0


class ExpertTrace:
    """Read-only routing capture for one loaded MoE model. Create it with :func:`trace`."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        *,
        run: str = "experts",
        root: str | None = None,
        name: str | None = None,
        model_id: str | None = None,
        tokens: bool = True,
        layers: Iterable[int] | None = None,
        store: bool = True,
        flush_every: float | None = 30.0,
        shard_tokens: int = 65_536,
    ) -> None:
        import torch

        if getattr(model, "_innards_experts", None) is not None:
            raise RuntimeError("This model is already traced; call trace.close() on the other ExpertTrace first")
        self.model = model
        self.tokenizer = tokenizer
        config = getattr(model, "config", None)
        self.model_id = (
            str(model_id or getattr(config, "_name_or_path", None) or getattr(model, "name_or_path", None) or "")
            or type(model).__name__
        )
        self.layers: list[MoELayer] = find_moe_layers(model, layers)
        if not self.layers:
            raise ValueError(
                f"No MoE layers found in {type(model).__name__}: Innards looks for modules with an `experts` "
                "submodule and two or more routed experts"
            )
        self.facts = routing_facts(model)
        self.device = next(model.parameters()).device
        self._float = torch.float32
        self._slots = {id(m): i for i, m in enumerate(self.layers)}
        self.run = safe_name(run or "experts")
        self.name = safe_name(name or self.model_id.rsplit("/", 1)[-1], "model")
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed and torch.distributed.get_world_size() > 1:
            self.name = f"{self.name}-rank{torch.distributed.get_rank()}"
        self.store = RunStore(root) if store else None
        self.flush_every = flush_every
        self.shard_tokens = shard_tokens

        # topics, device accumulators (added to between flushes) and CPU totals
        self.topics: list[str] = []
        self._topic_ids: dict[str, int] = {}
        self._label: Label = None
        self._keep = bool(tokens)
        self._default_keep = bool(tokens)
        self._rows: dict[tuple[int, str], Any] = {}
        self._capacity = 0
        self._counts: list[Any] = []
        self._gates: list[Any] = []
        self._seen: list[Any] = []
        self.totals: dict[str, Any] = {
            "counts": torch.zeros(0, len(self.layers), 0, dtype=torch.float64),
            "gates": torch.zeros(0, len(self.layers), 0, dtype=torch.float64),
            "tokens": torch.zeros(0, len(self.layers), dtype=torch.float64),
        }
        self._width = [max(m.experts, 1) for m in self.layers]
        self._buffer: list[_Forward] = []
        self._buffered = 0
        self._ctx: _Forward | None = None
        self._last: tuple[int, int] | None = None
        self._next_sequence = 0
        self._shards: list[str] = []
        self._vocab: dict[int, str] = {}
        self.forwards = 0
        self.overhead_ns = 0
        self.errors = 0
        self._logged: set[str] = set()
        self._closed = False
        self._last_flush = time.monotonic()
        self._started = now_iso()
        self._resume()

        self._handles: list[Any] = [
            model.register_forward_pre_hook(self._forward_start, with_kwargs=True),
            model.register_forward_hook(self._forward_end, always_call=True),
        ]
        attach(self.layers, self._on_route, self._register)
        model._innards_experts = self

    # ------------------------------------------------------------------ public API

    @property
    def folder(self) -> Path | None:
        """``<root>/<run>/experts/<name>``, where :meth:`flush` writes (``None`` with ``store=False``)."""
        return self.store.run_dir(self.run) / "experts" / self.name if self.store else None

    def topic(self, label: Label, *, tokens: bool | None = None) -> _Scope:
        """Label the forwards that follow: a topic for all of them, or one per sequence of a batch.

        ``trace.topic("math")`` sets it until changed; ``with trace.topic("math"):`` restores the previous
        label afterwards. ``None`` stops recording. ``tokens`` keeps (or skips) per-token routing for
        these forwards; the counts are always kept.
        """
        previous = (self._label, self._keep)
        labels = [label] if isinstance(label, str) else list(label) if label is not None else []
        if not all(isinstance(x, str) and x for x in labels):
            raise TypeError("A topic is a non-empty string, or a list of them (one per sequence in the batch)")
        for text in labels:
            self._topic_id(text)
        self._label = label if isinstance(label, str) or label is None else list(label)
        self._keep = self._default_keep if tokens is None else bool(tokens)
        self._rows = {}
        return _Scope(self, previous)

    def flush(self) -> Path | None:
        """Add the device counts to the totals and write the run folder. Waits for the device."""
        import torch

        start = time.perf_counter_ns()
        topics = len(self.topics)
        self._grow_totals(topics)  # every topic has a row, zero until something is measured
        if self._counts:
            for slot in range(len(self.layers)):
                counts, gates, seen = self._counts[slot], self._gates[slot], self._seen[slot]
                rows, width = min(topics, counts.shape[0]), counts.shape[1]
                # copy to the CPU first, then widen: on MPS, a row slice moved with .to("cpu", float64)
                # in one call arrives as zeros (torch 2.14)
                self.totals["counts"][:rows, slot, :width] += counts[:rows].cpu().to(torch.float64)
                self.totals["gates"][:rows, slot, :width] += gates[:rows].cpu().to(torch.float64)
                self.totals["tokens"][:rows, slot] += seen[:rows].cpu().to(torch.float64)
                counts.zero_()
                gates.zero_()
                seen.zero_()
        shard = self._write_shard() if self._buffer else None
        if shard is not None:
            self._shards.append(shard)
        self._last_flush = time.monotonic()
        self.overhead_ns += time.perf_counter_ns() - start
        if self.store is None:
            return None
        self.store.ensure_run(self.run, environment())
        folder = self.folder
        assert folder is not None
        _write_json(folder / "summary.json", self.summary())
        if self._vocab:
            _write_json(folder / "vocab.json", {str(k): v for k, v in sorted(self._vocab.items())})
        return folder

    def summary(self) -> dict[str, Any]:
        """Everything measured so far (as of the last :meth:`flush`), as JSON-ready data."""
        counts, gates, tokens = self.totals["counts"], self.totals["gates"], self.totals["tokens"]
        return {
            "format": FORMAT,
            "name": self.name,
            "run": self.run,
            "model": {
                "id": self.model_id,
                "class": type(self.model).__name__,
                "model_type": getattr(getattr(self.model, "config", None), "model_type", None),
                "routing": self.facts,
            },
            "layers": [m.to_dict() for m in self.layers],
            "topics": list(self.topics),
            "tokens": [[int(v) for v in row] for row in tokens.tolist()],
            "counts": [[[int(v) for v in layer] for layer in topic] for topic in counts.tolist()],
            "gates": [[[round(v, 6) for v in layer] for layer in topic] for topic in gates.tolist()],
            "forwards": self.forwards,
            "shards": list(self._shards),
            "health": {
                "overhead_ms": round(self.overhead_ns / 1e6, 3),
                "errors": self.errors,
                "missed": {str(m.layer): m.missed for m in self.layers if m.missed},
            },
            "started_at": self._started,
            "updated_at": now_iso(),
            "innards_version": __version__,
        }

    def map(self) -> Any:
        """An :class:`~innards.experts.ExpertMap` of everything measured (flushes first)."""
        from innards.experts.analysis import ExpertMap

        self.flush()
        return ExpertMap(self.summary(), folder=self.folder)

    def tokens(self) -> Any:
        """A :class:`~innards.experts.TokenRouting` over the token shards written so far (flushes first)."""
        from innards.experts.analysis import TokenRouting

        folder = self.flush()
        if folder is None:
            raise RuntimeError("Per-token routing is written to the run folder; this trace has store=False")
        return TokenRouting(folder)

    @property
    def last_sequence(self) -> int | None:
        """The sequence id of the most recent recorded forward (``None`` before the first one)."""
        return self._last[1] if self._last is not None else None

    def view(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        chat: bool | Mapping[str, Any] = False,
        topic: str = "chat",
        open_browser: bool = False,
        block: bool = False,
    ) -> Any:
        """Serve the web view over this trace's runs folder. Needs ``innards[view]``.

        ``chat=True`` (or a dict of ``generate()`` arguments, e.g. ``{"max_new_tokens": 256}``) lets the
        Conversation tab talk to this model: each message runs ``model.generate()`` inside
        ``trace.topic(...)``, with ``topic`` until the page names another. Without it, the page shows the
        traces and stored conversations only. Do not run the model from your own code while the page
        generates a reply.
        """
        from innards.view import serve

        if self.store is None:
            raise RuntimeError("The view reads the run folder; this trace has store=False")
        runner = None
        if chat:
            from innards.view.chat import ChatRunner

            runner = ChatRunner(self, generate=chat if isinstance(chat, Mapping) else None, topic=topic)
        self.flush()
        return serve(
            root=str(self.store.root), runner=runner, host=host, port=port, open_browser=open_browser, block=block
        )

    def close(self) -> None:
        """Remove every hook and write what was measured."""
        if self._closed:
            return
        self._closed = True
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        with contextlib.suppress(AttributeError):
            del self.model._innards_experts
        self._ctx = None
        self.flush()

    def __enter__(self) -> ExpertTrace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<ExpertTrace {self.model_id!r}: {len(self.layers)} MoE layers, topics {self.topics}>"

    # ------------------------------------------------------------------ hooks (generation / training path)

    def _register(
        self, module: Any, fn: Any, *, pre: bool = False, with_kwargs: bool = False, always_call: bool = False
    ) -> None:
        trace = self

        def wrapped(*args: Any) -> None:
            if trace._ctx is None:
                return None
            start = time.perf_counter_ns()
            try:
                fn(*args)
            except Exception:
                trace._error(type(args[0]).__name__)
            finally:
                trace.overhead_ns += time.perf_counter_ns() - start
            return None  # never replace a module's inputs or output

        if pre:
            self._handles.append(module.register_forward_pre_hook(wrapped, with_kwargs=with_kwargs))
        else:
            self._handles.append(module.register_forward_hook(wrapped, always_call=always_call))

    def _forward_start(self, _module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        if self._label is None or self._closed or self._ctx is not None:
            return None  # not labeled, closed, or a nested call of the same model
        start = time.perf_counter_ns()
        try:
            self._ctx = self._context(args, kwargs)
        except Exception:
            self._ctx = None
            self._error("forward")
        self.overhead_ns += time.perf_counter_ns() - start
        return None

    def _forward_end(self, _module: Any, _args: Any, _out: Any) -> None:
        ctx, self._ctx = self._ctx, None
        if ctx is None:
            return None
        start = time.perf_counter_ns()
        self.forwards += 1
        if ctx.keep and ctx.routes:
            self._buffer.append(ctx)
            self._buffered += ctx.n
        self.overhead_ns += time.perf_counter_ns() - start
        due = self.flush_every is not None and time.monotonic() - self._last_flush >= self.flush_every
        if due or self._buffered >= self.shard_tokens:
            try:
                self.flush()
            except Exception:
                self._error("flush")
        return None

    def _context(self, args: Any, kwargs: dict[str, Any]) -> _Forward | None:
        import torch

        ids = kwargs.get("input_ids")
        if ids is None and args and torch.is_tensor(args[0]) and not args[0].is_floating_point():
            ids = args[0]
        embeds = kwargs.get("inputs_embeds")
        ref = ids if ids is not None else embeds
        if not torch.is_tensor(ref):
            return None
        if ref.dim() == 1:
            batch, seq = 1, int(ref.shape[0])
        else:
            batch, seq = int(ref.shape[0]), int(ref.shape[1])
        rows = self._topic_rows(batch, ref.device)
        if rows is None:
            return None
        n, device = batch * seq, ref.device
        past = _past_tokens(kwargs)
        mask = kwargs.get("attention_mask")
        if torch.is_tensor(mask) and mask.dim() == 2 and mask.shape[0] == batch and mask.shape[1] >= seq:
            valid = (mask[:, -seq:] != 0).reshape(-1)
            position = (mask.long().cumsum(-1) - 1)[:, -seq:].reshape(-1)
        else:
            valid = torch.ones(n, dtype=torch.bool, device=device)
            position = (torch.arange(seq, device=device) + past).repeat(batch)
        if past > 0 and self._last is not None and self._last[0] == batch:
            base = self._last[1]  # a decode step: the same sequences as the forward before
        else:
            base = self._next_sequence
            self._next_sequence += batch
        self._last = (batch, base)
        sequence = (torch.arange(batch, device=device) + base).repeat_interleave(seq)
        token = (
            ids.reshape(-1).to(torch.int32)
            if ids is not None
            else torch.full((n,), -1, dtype=torch.int32, device=device)
        )
        return _Forward(n, valid, rows.repeat_interleave(seq), token, sequence, position, self._keep)

    def _topic_rows(self, batch: int, device: Any) -> Any:
        import torch

        label = self._label
        key = (batch, str(device))
        if key in self._rows:
            return self._rows[key]
        if isinstance(label, str):
            ids = [self._topic_ids[label]] * batch
        elif isinstance(label, list) and len(label) == batch:
            ids = [self._topic_ids[x] for x in label]
        else:
            n = len(label) if isinstance(label, list) else 0
            if "labels" not in self._logged:
                self._logged.add("labels")
                log.warning("innards: %d topic labels for a batch of %d sequences; forward not recorded", n, batch)
            self.errors += 1
            return None
        rows = self._rows[key] = torch.tensor(ids, dtype=torch.long, device=device)
        return rows

    def _on_route(self, moe: MoELayer, index: Any, weights: Any) -> None:
        import torch

        ctx = self._ctx
        if ctx is None:
            return
        slot = self._slots[id(moe)]
        k = int(index.shape[1])
        if int(index.shape[0]) != ctx.n:
            moe.missed += 1  # rows that do not line up with the batch (e.g. modality-split experts)
            if f"rows-{slot}" not in self._logged:
                self._logged.add(f"rows-{slot}")
                log.warning(
                    "innards: layer %d routed %d rows for %d tokens; not recorded", moe.layer, index.shape[0], ctx.n
                )
            return
        width = self._width[slot]
        index = index.long()
        slot_ok = (index >= 0) & (index < width)  # sentinel slots (zero-compute experts) are not experts
        if weights is not None:
            weights = weights.to(self._float)
            slot_ok &= weights != 0  # a slot with weight exactly 0 is a skipped slot, not an activation
            share = weights / weights.abs().sum(-1, keepdim=True).clamp_min(1e-20)
        else:
            share = torch.full(index.shape, 1.0 / k, dtype=self._float, device=index.device)
        on = ctx.valid[:, None] & slot_ok
        flat = (ctx.topic[:, None] * width + index.clamp(0, width - 1)).reshape(-1)
        self._ensure(slot, index.device)
        self._counts[slot].view(-1).index_add_(0, flat, on.reshape(-1).to(self._float))
        self._gates[slot].view(-1).index_add_(0, flat, (share * on).reshape(-1))
        self._seen[slot].index_add_(0, ctx.topic, ctx.valid.to(self._float))
        if ctx.keep:
            experts = torch.where(slot_ok, index, torch.full_like(index, -1)).to(torch.int16)
            nan = torch.full_like(share, float("nan"))
            kept = weights.to(torch.float32, copy=True) if weights is not None else nan
            ctx.routes[slot] = (experts, kept)

    # ------------------------------------------------------------------ bookkeeping

    def _topic_id(self, label: str) -> int:
        if label not in self._topic_ids:
            self._topic_ids[label] = len(self.topics)
            self.topics.append(label)
        return self._topic_ids[label]

    def _ensure(self, slot: int, device: Any) -> None:
        """Device accumulators with room for every topic registered so far."""
        import torch

        need = len(self.topics)
        if not self._counts:
            self._float = torch.float32
            self._counts = [None] * len(self.layers)
            self._gates = [None] * len(self.layers)
            self._seen = [None] * len(self.layers)
        current = self._counts[slot]
        if current is not None and current.shape[0] >= need:
            return
        size = max(need, 4, 2 * (current.shape[0] if current is not None else 0))
        width = self._width[slot]
        grown = [
            torch.zeros(size, width, dtype=self._float, device=device),
            torch.zeros(size, width, dtype=self._float, device=device),
            torch.zeros(size, dtype=self._float, device=device),
        ]
        if current is not None:
            rows = current.shape[0]
            grown[0][:rows] = self._counts[slot]
            grown[1][:rows] = self._gates[slot]
            grown[2][:rows] = self._seen[slot]
        self._counts[slot], self._gates[slot], self._seen[slot] = grown
        for other in range(len(self.layers)):  # every layer keeps the same topic rows
            if other != slot and self._counts[other] is None:
                self._ensure(other, device)

    def _grow_totals(self, topics: int) -> None:
        import torch

        width = max(self._width)
        for key in ("counts", "gates"):
            old = self.totals[key]
            if old.shape[0] < topics or old.shape[2] < width:
                new = torch.zeros(topics, len(self.layers), width, dtype=torch.float64)
                new[: old.shape[0], :, : old.shape[2]] = old
                self.totals[key] = new
        old = self.totals["tokens"]
        if old.shape[0] < topics:
            new = torch.zeros(topics, len(self.layers), dtype=torch.float64)
            new[: old.shape[0]] = old
            self.totals["tokens"] = new

    def _error(self, where: str) -> None:
        self.errors += 1
        if where not in self._logged:  # log the first failure of each kind, count the rest
            self._logged.add(where)
            log.warning("innards: expert routing hook failed in %s; the model runs on", where, exc_info=True)

    def _write_shard(self) -> str | None:
        """Write the buffered per-token routing as one safetensors shard (valid tokens only)."""
        import torch

        forwards, self._buffer, self._buffered = self._buffer, [], 0
        width = max((r[0].shape[1] for f in forwards for r in f.routes.values()), default=1)
        layers = len(self.layers)
        parts: dict[str, list[Any]] = {k: [] for k in ("experts", "weights", "token", "sequence", "position", "topic")}
        for f in forwards:
            experts = torch.full((f.n, layers, width), -1, dtype=torch.int16, device=f.valid.device)
            weights = torch.full((f.n, layers, width), float("nan"), dtype=torch.float32, device=f.valid.device)
            for slot, (e, w) in f.routes.items():
                experts[:, slot, : e.shape[1]] = e
                weights[:, slot, : w.shape[1]] = w
            keep = f.valid
            parts["experts"].append(experts[keep].cpu())
            parts["weights"].append(weights[keep].cpu())
            parts["token"].append(f.token[keep].cpu())
            parts["sequence"].append(f.sequence[keep].cpu())
            parts["position"].append(f.position[keep].to(torch.int32).cpu())
            parts["topic"].append(f.topic[keep].to(torch.int16).cpu())
        tensors = {k: torch.cat(v).contiguous() for k, v in parts.items()}
        if not tensors["token"].numel():
            return None
        self._decode_new(tensors["token"])
        if self.store is None:
            return None
        from safetensors.torch import save_file

        folder = self.folder
        assert folder is not None
        (folder / "tokens").mkdir(parents=True, exist_ok=True)
        relative = f"tokens/shard-{len(self._shards) + 1:06d}.safetensors"
        meta = {
            "format": TOKENS_FORMAT,
            "layers": json.dumps([m.layer for m in self.layers]),
            "topics": json.dumps(self.topics),
            "model": self.model_id,
        }
        save_file(tensors, str(folder / relative), metadata=meta)
        return relative

    def _decode_new(self, ids: Any) -> None:
        if self.tokenizer is None:
            return
        new = sorted({int(i) for i in ids.unique().tolist() if i >= 0} - self._vocab.keys())
        if not new:
            return
        try:
            texts = self.tokenizer.batch_decode([[i] for i in new])
        except Exception:
            texts = [str(i) for i in new]
        self._vocab.update(zip(new, (str(t) for t in texts), strict=True))

    def _resume(self) -> None:
        """Continue a trace folder written earlier for the same model (same run and name)."""
        import torch

        folder = self.folder
        if folder is None or not (folder / "summary.json").is_file():
            return
        data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if (data.get("model") or {}).get("id") != self.model_id:
            raise ValueError(
                f"{folder} holds a trace of {(data.get('model') or {}).get('id')!r}, not {self.model_id!r}; "
                "pass another name= or run="
            )
        if [m["layer"] for m in data.get("layers") or []] != [m.layer for m in self.layers]:
            raise ValueError(f"{folder} was traced with other MoE layers; pass another name= or run=")
        for label in data.get("topics") or []:
            self._topic_id(str(label))
        if self.topics:
            self.totals["counts"] = torch.tensor(data["counts"], dtype=torch.float64)
            self.totals["gates"] = torch.tensor(data["gates"], dtype=torch.float64)
            self.totals["tokens"] = torch.tensor(data["tokens"], dtype=torch.float64)
        self._shards = list(data.get("shards") or [])
        self.forwards = int(data.get("forwards") or 0)
        self._started = data.get("started_at") or self._started
        vocab = folder / "vocab.json"
        if vocab.is_file():
            self._vocab = {int(k): v for k, v in json.loads(vocab.read_text(encoding="utf-8")).items()}
        last = max((int(Path(s).stem.rsplit("-", 1)[-1]) for s in self._shards), default=0)
        if last:
            import safetensors

            with safetensors.safe_open(str(folder / self._shards[-1]), "pt") as fh:
                self._next_sequence = int(fh.get_tensor("sequence").max()) + 1


def _write_json(path: Path, data: dict[str, Any]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)  # atomic: the view never reads a half-written file


def trace(
    model: Any,
    tokenizer: Any = None,
    *,
    run: str = "experts",
    root: str | None = None,
    name: str | None = None,
    model_id: str | None = None,
    tokens: bool = True,
    layers: Iterable[int] | None = None,
    store: bool = True,
    flush_every: float | None = 30.0,
    shard_tokens: int = 65_536,
) -> ExpertTrace:
    """Attach read-only hooks to every MoE layer of a model you loaded; record routing by topic.

    Works during inference (``generate``) and training (any forward of the model), on any MoE
    architecture with an ``experts`` submodule, including remote code (see :mod:`innards.experts.routing`).

    Args:
        model: The loaded model, the module your code calls (``AutoModelForCausalLM`` ...).
        tokenizer: Its tokenizer, so tokens show as text (optional).
        run: Run name. Results go to ``<root>/<run>/experts/<name>/``.
        root: Folder of runs (default ``$INNARDS_RUNS`` or ``./innards_runs``).
        name: Name of this trace in the run (default: the model's name).
        tokens: Keep every token's chosen experts and weights (token shards). ``False`` keeps the
            topic x layer x expert counts only, the lighter choice for long training runs.
        layers: Hook these decoder layer indices only.
        store: ``False`` keeps everything in memory (``trace.map()`` still works; no token shards).
        flush_every: Seconds between automatic flushes (``None``: only on :meth:`~ExpertTrace.flush`).
        shard_tokens: Write a token shard once this many token rows are buffered.
    """
    return ExpertTrace(
        model,
        tokenizer,
        run=run,
        root=root,
        name=name,
        model_id=model_id,
        tokens=tokens,
        layers=layers,
        store=store,
        flush_every=flush_every,
        shard_tokens=shard_tokens,
    )
