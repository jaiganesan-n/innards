"""Find the MoE layers of a loaded model and read which experts each token was sent to, read-only.

Every MoE block routes tokens in its own code, so a layer is read from the first of three places
that carries the model's own decision:

* ``experts_input``: the block calls its experts module as ``experts(hidden, top_k_index, top_k_weights)``
  (transformers 5). A forward pre-hook on that module sees exactly the experts and weights each
  token was sent to, whatever the router did before (bias correction, group-limited routing ...).
* ``router_output``: the router module returns the chosen experts, e.g. ``(logits, weights, indices)``.
* ``topk``: anything else, such as remote code that routes inline and never calls a router or
  experts module (``facebook/MobileMoE``). While the block runs, a torch-function mode watches the
  block's own ``torch.topk`` over the experts and takes the tensors it returned; a ``gather`` that
  reuses those indices gives the unbiased weights. The mode is removed at the first submodule call
  after the selection, so the expert computation itself runs untouched.

A layer's first forward decides its source (exact module hooks win) and records it with the data.
Nothing here changes a module's inputs or outputs: hooks return ``None`` and the mode returns the
result of the call it observed.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MoELayer", "Register", "Route", "attach", "find_moe_layers", "routing_facts"]

_LAYER = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)(?:\.|$)")
_ANY_INDEX = re.compile(r"\.(\d+)(?:\.|$)")
_ROUTER_ATTRS = ("gate", "router")
_SHARED_ATTRS = ("shared_experts", "shared_expert", "shared_mlp")
_EXPERT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
_TOPK_KEYS = ("num_experts_per_tok", "moe_topk", "top_k", "num_selected_experts")
_SHARED_KEYS = ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts")
_ROUTING_KEYS = (
    "norm_topk_prob",
    "scoring_func",
    "routed_scaling_factor",
    "n_group",
    "topk_group",
    "topk_method",
    "router_jitter_noise",
    "moe_latent_size",
)

Register = Callable[..., None]
"""``register(module, fn, *, pre=False, with_kwargs=False, always_call=False)``: add a read-only hook.

The caller wraps ``fn`` (gating, timing, fail-open) and registers it on ``module``.
"""

Route = Callable[["MoELayer", Any, Any], None]
"""``on_route(layer, indices[tokens, k], weights[tokens, k] | None)``, once per block forward."""


@dataclass
class MoELayer:
    """One MoE block of a loaded model and where its routing is read."""

    layer: int
    """Decoder layer index (from the module path)."""
    name: str
    """Module path of the MoE block, e.g. ``model.layers.3.mlp``."""
    experts: int
    """Routed experts in this block."""
    top_k: int | None
    experts_module: str | None = None
    router_module: str | None = None
    shared_module: str | None = None
    candidates: tuple[str, ...] = ()
    """Sources to try, in order (``experts_input``, ``router_output``, ``topk``)."""
    source: str | None = None
    """The source the first forward settled on."""
    weights: str | None = None
    """What the recorded weights are: ``applied`` (what the experts were scaled by), ``gather``
    (the unbiased scores a ``gather`` took for the chosen experts) or ``topk_values`` (the scores
    ``torch.topk`` selected on, which can include a selection bias)."""
    missed: int = 0
    """Block forwards where no source produced routing."""
    block: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "block": self.name,
            "experts": self.experts,
            "top_k": self.top_k,
            "source": self.source,
            "weights": self.weights,
            "experts_module": self.experts_module,
            "router_module": self.router_module,
            "shared_module": self.shared_module,
            "missed": self.missed,
        }


def _config(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    data: dict[str, Any] = to_dict() if callable(to_dict) else dict(config or {})
    for key in ("text_config", "language_config", "llm_config"):
        sub = data.get(key)
        if isinstance(sub, Mapping) and sub:
            return dict(sub)
    return data


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _first(cfg: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        value = _int(cfg.get(key))
        if value is not None:
            return value
    return None


def routing_facts(model_or_config: Any) -> dict[str, Any]:
    """Routing settings the config declares (read from the config, not measured)."""
    cfg = dict(model_or_config) if isinstance(model_or_config, Mapping) else _config(model_or_config)
    facts: dict[str, Any] = {
        "experts": _first(cfg, _EXPERT_KEYS),
        "top_k": _first(cfg, _TOPK_KEYS),
        "shared_experts": _first(cfg, _SHARED_KEYS),
    }
    for key in _ROUTING_KEYS:
        if cfg.get(key) is not None:
            facts[key] = cfg[key]
    return facts


def layer_of(name: str) -> int | None:
    match = _LAYER.search(name) or _ANY_INDEX.search(name)
    return int(match.group(1)) if match else None


def _module(owner: Any, attrs: Sequence[str]) -> tuple[str | None, Any]:
    import torch

    for attr in attrs:
        value = getattr(owner, attr, None)
        if isinstance(value, torch.nn.Module):
            return attr, value
    return None, None


def _calls(block: Any, attr: str | None) -> bool:
    """Whether the block's forward source calls ``self.<attr>(``; unknown source counts as yes."""
    if attr is None:
        return False
    try:
        source = inspect.getsource(type(block).forward)
    except (OSError, TypeError):
        return True
    return re.search(rf"self\.{re.escape(attr)}\s*\(", source) is not None


def _has_forward(module: Any) -> bool:
    import torch

    return type(module).forward is not torch.nn.Module.forward


def _count_experts(block: Any, experts: Any, router: Any, cfg: Mapping[str, Any]) -> int | None:
    import torch

    for owner in (experts, block, router):
        for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
            value = _int(getattr(owner, attr, None))
            if value is not None:
                return value
    if isinstance(experts, (torch.nn.ModuleList, torch.nn.Sequential)) and len(experts) > 1:
        return len(experts)
    weight = getattr(router, "weight", None)
    if torch.is_tensor(weight) and weight.dim() == 2 and weight.shape[0] > 1:
        return int(weight.shape[0])
    return _first(cfg, _EXPERT_KEYS)


def _count_top_k(block: Any, router: Any, cfg: Mapping[str, Any]) -> int | None:
    for owner in (block, router):
        for attr in ("top_k", "num_experts_per_tok", "topk"):
            value = _int(getattr(owner, attr, None))
            if value is not None:
                return value
    return _first(cfg, _TOPK_KEYS)


def find_moe_layers(model: Any, layers: Iterable[int] | None = None) -> list[MoELayer]:
    """Every MoE block in ``model``: a module with an ``experts`` submodule and a router or expert count.

    Works on any ``torch.nn.Module`` (transformers classes and remote code alike); ``layers`` keeps
    only those decoder layer indices.
    """
    import torch

    cfg = _config(model)
    wanted = set(layers) if layers is not None else None
    found: list[MoELayer] = []
    inside: list[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(prefix + ".") for prefix in inside):
            continue  # an MoE block's own submodules are not blocks
        experts = getattr(module, "experts", None)
        if not isinstance(experts, torch.nn.Module) or not name:
            continue
        router_attr, router = _module(module, _ROUTER_ATTRS)
        count = _count_experts(module, experts, router, cfg)
        if count is None or count < 2:
            continue
        inside.append(name)
        layer = layer_of(name)
        if layer is None:
            layer = len(found)
        if wanted is not None and layer not in wanted:
            continue
        shared_attr, _ = _module(module, _SHARED_ATTRS)
        candidates = []
        if _has_forward(experts) and _calls(module, "experts"):
            candidates.append("experts_input")
        if router is not None and _has_forward(router) and _calls(module, router_attr):
            candidates.append("router_output")
        candidates.append("topk")
        found.append(
            MoELayer(
                layer=layer,
                name=name,
                experts=count,
                top_k=_count_top_k(module, router, cfg),
                experts_module=f"{name}.experts",
                router_module=f"{name}.{router_attr}" if router_attr else None,
                shared_module=f"{name}.{shared_attr}" if shared_attr else None,
                candidates=tuple(candidates),
                block=module,
            )
        )
    return found


# --------------------------------------------------------------------------- reading the routing


def _is_index(t: Any, top_k: int | None) -> bool:
    import torch

    return (
        torch.is_tensor(t)
        and t.dim() >= 1
        and not t.is_floating_point()
        and t.dtype != torch.bool
        and (top_k is None or t.shape[-1] == top_k)
    )


def _pick(tensors: Sequence[Any], top_k: int | None) -> tuple[Any, Any]:
    """The chosen-experts tensor and a float tensor of the same shape (its weights), from candidates."""
    import torch

    index = next((t for t in tensors if _is_index(t, top_k)), None)
    if index is None:
        return None, None
    weights = next(
        (t for t in tensors if torch.is_tensor(t) and t.is_floating_point() and t.shape == index.shape), None
    )
    return index, weights


def _flat(index: Any, weights: Any) -> tuple[Any, Any]:
    k = int(index.shape[-1]) if index.dim() >= 2 else 1
    index = index.detach().reshape(-1, k)
    if weights is not None:
        weights = weights.detach().reshape(-1, k)
    return index, weights


@dataclass
class _Step:
    """Routing seen during one forward of one block."""

    experts: list[tuple[Any, Any]] = field(default_factory=list)
    router: tuple[Any, Any] | None = None
    watching: bool = False
    """The top-k mode was active for this forward."""


def _cat(parts: list[tuple[Any, Any]]) -> tuple[Any, Any]:
    """One ``(indices, weights)`` from the experts calls of one block forward (usually exactly one)."""
    import torch

    if len(parts) == 1:
        return parts[0]
    weights = [w for _, w in parts]
    joined = torch.cat(weights) if all(w is not None for w in weights) else None
    return torch.cat([i for i, _ in parts]), joined


class _TopK:
    """A torch-function mode, active only inside one MoE block, that keeps the block's expert selection."""

    def __init__(self, experts: int, top_k: int | None) -> None:
        import torch
        from torch.overrides import TorchFunctionMode

        topk_fns = {torch.topk, torch.Tensor.topk}
        gather_fns = {torch.gather, torch.Tensor.gather}
        owner = self
        self.experts, self.top_k = experts, top_k
        self.indices: Any = None
        self.values: Any = None
        self.gathered: Any = None
        self.pushed = False

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                if func in topk_fns:
                    owner._topk(args, kwargs, out)
                elif func in gather_fns and owner.indices is not None:
                    index = args[2] if len(args) > 2 else kwargs.get("index")
                    if index is owner.indices:
                        owner.gathered = out
                return out

        self._mode = Mode()

    def _topk(self, args: Any, kwargs: Mapping[str, Any], out: Any) -> None:
        import torch

        x = args[0] if args else kwargs.get("input")
        k = args[1] if len(args) > 1 else kwargs.get("k")
        dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
        if not torch.is_tensor(x) or x.dim() < 2 or dim not in (-1, x.dim() - 1) or x.shape[-1] != self.experts:
            return
        if self.top_k is not None and k != self.top_k:
            return
        self.values, self.indices, self.gathered = out[0], out[1], None  # the last selection wins

    def push(self) -> None:
        self.indices = self.values = self.gathered = None
        self._mode.__enter__()  # type: ignore[no-untyped-call]
        self.pushed = True

    def pop(self) -> None:
        if self.pushed:
            self.pushed = False
            self._mode.__exit__(None, None, None)  # type: ignore[no-untyped-call]

    def result(self) -> tuple[Any, Any, str] | None:
        if self.indices is None:
            return None
        if self.gathered is not None and self.gathered.shape == self.indices.shape:
            return self.indices, self.gathered, "gather"
        return self.indices, self.values, "topk_values"


def _children(block: Any) -> list[Any]:
    """Direct submodules and theirs (not into ModuleLists of experts): where the expert work starts."""
    import torch

    out = []
    for child in block.children():
        out.append(child)
        if not isinstance(child, (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)):
            out.extend(c for c in child.children() if not isinstance(c, torch.nn.ModuleList))
    return out


def attach(layers: Sequence[MoELayer], on_route: Route, register: Register) -> None:
    """Hook every layer so ``on_route(layer, indices, weights)`` runs once per block forward.

    ``register`` adds each hook with the caller's gating, timing and fail-open wrapper; when a
    hook is gated off it never runs, so a block forward either sees all of its hooks or none.
    """
    for moe in layers:
        _attach_one(moe, on_route, register)


def _attach_one(moe: MoELayer, on_route: Route, register: Register) -> None:
    block = moe.block
    state: dict[str, Any] = {"step": None}
    watcher = _TopK(moe.experts, moe.top_k) if "topk" in moe.candidates else None

    def start(_m: Any, _args: Any) -> None:
        step = state["step"] = _Step()
        if watcher is not None and moe.source in (None, "topk"):
            watcher.push()
            step.watching = True

    def child_start(_m: Any, _args: Any) -> None:
        if watcher is not None and watcher.pushed and watcher.indices is not None:
            watcher.pop()  # the selection is made; the expert computation runs without the mode

    def experts_in(_m: Any, args: Any, kwargs: Mapping[str, Any]) -> None:
        step = state["step"]
        if step is None:
            return
        tensors = list(args[1:]) + [v for k, v in kwargs.items() if k != "hidden_states"]
        index, weights = _pick(tensors, moe.top_k)
        if index is not None:
            step.experts.append(_flat(index, weights))

    def router_out(_m: Any, _args: Any, out: Any) -> None:
        step = state["step"]
        if step is None:
            return
        items = list(out) if isinstance(out, (tuple, list)) else [out]
        index, weights = _pick(items, moe.top_k)
        if index is not None:
            step.router = _flat(index, weights)

    def end(_m: Any, _args: Any, _out: Any) -> None:
        step, state["step"] = state["step"], None
        if watcher is not None:
            watcher.pop()
        if step is None:
            return
        topk = watcher.result() if watcher is not None and step.watching else None
        chosen: tuple[Any, Any] | None = None
        if step.experts and moe.source in (None, "experts_input"):
            chosen, source, weights = _cat(step.experts), "experts_input", "applied"
        elif step.router is not None and moe.source in (None, "experts_input", "router_output"):
            chosen, source, weights = step.router, "router_output", "applied"
        elif topk is not None:
            chosen, source, weights = _flat(topk[0], topk[1]), "topk", topk[2]
        if chosen is None:
            moe.missed += 1
            moe.source = None  # try every source again on the next forward
            return
        moe.source, moe.weights = source, weights
        on_route(moe, chosen[0], chosen[1])

    register(block, start, pre=True)
    register(block, end, always_call=True)
    if "experts_input" in moe.candidates:
        register(block.experts, experts_in, pre=True, with_kwargs=True)
    if "router_output" in moe.candidates and moe.router_module is not None:
        register(getattr(block, moe.router_module.rsplit(".", 1)[-1]), router_out)
    if watcher is not None:
        for child in _children(block):
            register(child, child_start, pre=True)
