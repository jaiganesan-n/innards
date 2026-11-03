"""Records Innards writes: one TurnRecord per ``generate()`` call (schema 1.0).

A **run** groups conversations under one name (``observe(..., run="deltanet-study")``) and is one
folder on disk. A **conversation** is one model talking through a sequence of turns. A **turn** is one
``generate()`` call. Each turn carries the light tier (tokens, context, KV, memory, timing) and one
payload per architectural feature that was analyzed (``features["gated_deltanet"]`` ...).

Rules:

* Missing measurements are ``None`` (JSON ``null``), never zero.
* ``kv.measured_source`` names where a measured value came from (``hf_cache``).
* Breaking changes bump the major version. Readers accept 1.x and reject other majors.
  New optional fields are added within 1.x.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

__all__ = [
    "SCHEMA_VERSION",
    "ContextComposition",
    "ConversationRecord",
    "KVInfo",
    "MemoryInfo",
    "ModelInfo",
    "ObserverInfo",
    "OutputInfo",
    "ProbeHealth",
    "RuntimeInfo",
    "TimingInfo",
    "TokenCounts",
    "TurnRecord",
]

SCHEMA_VERSION = "1.0"
_READABLE_MAJORS = {"1"}

_T = TypeVar("_T")


def _build(cls: type[_T], data: Any) -> _T:
    """Construct a dataclass from a dict, ignoring unknown keys (forward compatible within 1.x)."""
    if isinstance(data, cls):
        return data
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


def _check_version(data: dict[str, Any], kind: str) -> str:
    version = str(data.get("schema_version", SCHEMA_VERSION))
    if version.split(".")[0] not in _READABLE_MAJORS:
        raise ValueError(f"Unsupported {kind} schema {version}; this reader handles 1.x")
    return version


@dataclass(slots=True)
class ModelInfo:
    id: str
    architecture: str | None = None
    """Lineage family: mha, mqa, gqa, mla, dsa, compressed, sliding_window, hybrid or ssm."""
    attention: str | None = None
    """Head design of the attention layers: mha, mqa, gqa or mla."""
    lineage: dict[str, Any] | None = None
    modalities: list[str] = field(default_factory=lambda: ["text"])
    layers: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    sliding_window: int | None = None
    dtype: str | None = None
    """Compute dtype of the weights as loaded."""
    kv_dtype: str | None = None
    quantization: str | None = None
    """The checkpoint's own published quantization (e.g. ``"fp8"`` for DeepSeek-V4); ``None`` for plain weights."""
    model_type: str | None = None
    features: list[str] = field(default_factory=list)
    """Architectural features detected from the config (``gated_deltanet``, ``mhc``, ``moe`` ...)."""
    parameters: int | None = None


@dataclass(slots=True)
class RuntimeInfo:
    backend: str
    """How the turn was produced: ``observe`` (hooks on the model your code loaded)."""
    host: str | None = None
    device: str | None = None
    python: str | None = None
    engine_version: str | None = None
    innards_version: str | None = None


@dataclass(slots=True)
class TokenCounts:
    prompt: int | None = None
    """Prompt tokens for this turn (the whole input to ``generate``, including any cached prefix)."""
    cached: int | None = None
    """Prompt tokens already in the cache passed to ``generate`` (prefix reuse)."""
    recomputed: int | None = None
    """Prompt tokens prefilled this turn (``prompt - cached``)."""
    generated: int | None = None
    context_after: int | None = None
    """``prompt + generated``."""
    by_modality: dict[str, int] | None = None
    batch_size: int | None = None


@dataclass(slots=True)
class ContextComposition:
    system: int | None = None
    history: int | None = None
    summary: int | None = None
    new_message: int | None = None


@dataclass(slots=True)
class KVInfo:
    predicted_bytes: int | None = None
    """Architectural KV (plus recurrent state when known) for ``cache_tokens`` resident tokens."""
    measured_bytes: int | None = None
    measured_source: str | None = None
    by_layer_type: dict[str, int] | None = None
    """Measured bytes split into ``global``, ``sliding`` and ``recurrent`` (plus ``compressed``,
    ``indexer`` and ``pending`` for models that have them)."""
    per_layer_bytes: list[int] | None = None
    prefix_reused: bool | None = None
    capacity_tokens: int | None = None
    turns_to_limit: int | None = None
    cache_tokens: int | None = None
    """Tokens resident in the cache after the turn. With HF ``generate`` this is
    ``context_after - 1``: the last sampled token has not been fed back yet."""
    predicted_by_layer_type: dict[str, int] | None = None
    predicted_per_layer_bytes: list[int] | None = None
    growth_bytes: int | None = None
    """Measured KV change since the previous turn of the conversation (the whole cache on turn 1)."""


@dataclass(slots=True)
class MemoryInfo:
    weights_bytes: int | None = None
    kv_bytes: int | None = None
    activations_bytes: int | None = None
    """``peak - weights - kv``; only on GPU, where the allocator counts tensors alone."""
    peak_bytes: int | None = None
    allocated_bytes: int | None = None
    """Allocated after the turn (GPU allocator), or process RSS on CPU."""
    budget_bytes: int | None = None
    """The memory budget Innards checks against (16 GB by default)."""
    source: str | None = None
    """``torch_cuda`` (allocator counters), ``torch_mps`` or ``psutil_rss`` (process RSS)."""


@dataclass(slots=True)
class TimingInfo:
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_tokens_per_s: float | None = None
    total_ms: float | None = None


@dataclass(slots=True)
class OutputInfo:
    text: str | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class ProbeHealth:
    """What one feature probe cost and whether it lost anything, for this turn."""

    feature: str
    hooks: int = 0
    overhead_ms: float | None = None
    """Host time spent inside this probe's hooks during the turn."""
    overhead_pct: float | None = None
    """``overhead_ms`` as a share of the turn's generation time."""
    sampled_tokens: int | None = None
    dropped: int = 0
    errors: int = 0


@dataclass(slots=True)
class ObserverInfo:
    dropped_events: int = 0
    """Events dropped by the bounded queue so far in this conversation."""
    errors: int = 0
    """Observer or analyzer errors so far in this conversation (generation continued)."""
    overhead_ms: float | None = None
    """Host time Innards spent on the generation thread during this turn (hooks and taps)."""
    overhead_pct: float | None = None
    """``overhead_ms`` as a share of the turn's total time. The budget is 1%."""
    probes: list[ProbeHealth] = field(default_factory=list)


@dataclass(slots=True)
class TurnRecord:
    """Everything Innards records about one ``generate()`` call."""

    conversation: str
    turn: int
    timestamp: str
    model: ModelInfo
    runtime: RuntimeInfo
    run: str | None = None
    message: str | None = None
    """The new user input of this turn (as given to the runner, or decoded from the new tokens)."""
    set: dict[str, Any] | None = None
    """Conversation-set step that produced the message: ``{"name", "step", "of"}``."""
    tokens: TokenCounts = field(default_factory=TokenCounts)
    context: ContextComposition = field(default_factory=ContextComposition)
    kv: KVInfo = field(default_factory=KVInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    timing: TimingInfo = field(default_factory=TimingInfo)
    output: OutputInfo = field(default_factory=OutputInfo)
    observer: ObserverInfo = field(default_factory=ObserverInfo)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    """One payload per analyzed feature: ``{"summary": {...}, "findings": [...], "data": {...}}``."""
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        ordered = {"schema_version": data.pop("schema_version")}
        ordered.update(data)
        return ordered

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnRecord:
        version = _check_version(data, "TurnRecord")
        conversation = data.get("conversation")
        if conversation is None:
            raise ValueError("TurnRecord needs `conversation`")
        observer = dict(data.get("observer") or {})
        observer["probes"] = [_build(ProbeHealth, p) for p in observer.get("probes") or []]
        return cls(
            schema_version=version,
            run=data.get("run"),
            conversation=str(conversation),
            turn=data["turn"],
            timestamp=data["timestamp"],
            message=data.get("message"),
            set=data.get("set"),
            model=_build(ModelInfo, data.get("model")),
            runtime=_build(RuntimeInfo, data.get("runtime")),
            tokens=_build(TokenCounts, data.get("tokens")),
            context=_build(ContextComposition, data.get("context")),
            kv=_build(KVInfo, data.get("kv")),
            memory=_build(MemoryInfo, data.get("memory")),
            timing=_build(TimingInfo, data.get("timing")),
            output=_build(OutputInfo, data.get("output")),
            observer=_build(ObserverInfo, observer),
            features=dict(data.get("features") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> TurnRecord:
        return cls.from_dict(json.loads(text))


@dataclass(slots=True)
class ConversationRecord:
    """One conversation: a model and the turns it took. Stored as ``<run>/<id>/conversation.json``."""

    id: str
    model: ModelInfo
    run: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    status: str = "active"
    """``active``, ``completed``, ``interrupted`` or ``failed``."""
    turns: int = 0
    set: str | None = None
    """Conversation set this conversation ran (``with_context`` ...), if any."""
    label: str | None = None
    runtime: RuntimeInfo | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    feature_status: dict[str, str] = field(default_factory=dict)
    """Per feature: ``analyzed``, ``detected``, ``not_analyzed`` or ``not_run``
    (see :func:`innards.features.feature_status`)."""
    config: dict[str, Any] = field(default_factory=dict)
    """The model's config, so a copied run folder can be analyzed and compared offline."""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        ordered = {"schema_version": data.pop("schema_version")}
        ordered.update(data)
        return ordered

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationRecord:
        _check_version(data, "ConversationRecord")
        runtime = data.get("runtime")
        return cls(
            id=str(data["id"]),
            model=_build(ModelInfo, data.get("model")),
            run=data.get("run"),
            started_at=data.get("started_at"),
            updated_at=data.get("updated_at"),
            status=str(data.get("status") or "active"),
            turns=int(data.get("turns") or 0),
            set=data.get("set"),
            label=data.get("label"),
            runtime=_build(RuntimeInfo, runtime) if runtime else None,
            environment=dict(data.get("environment") or {}),
            settings=dict(data.get("settings") or {}),
            feature_status=dict(data.get("feature_status") or {}),
            config=dict(data.get("config") or {}),
        )
