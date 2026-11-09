"""Load a model config and normalize it into a :class:`ModelSpec`.

Pure Python: no torch, no transformers. The layer-pattern rules mirror how
transformers builds its cache (``layer_types``), so predictions line up with
what the HF backend measures.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ConfigSource",
    "LayerKV",
    "ModelSpec",
    "load_config",
    "spec_from_config",
]

ConfigSource = str | os.PathLike[str] | Mapping[str, Any]

# Canonical per-layer cache types (a subset of transformers' `layer_types` names).
GLOBAL = "full_attention"
SLIDING = "sliding_attention"
CHUNKED = "chunked_attention"
LINEAR = "linear_attention"  # recurrent / SSM / conv state, fixed size
HYBRID = "hybrid"  # attention KV + recurrent state in the same layer
HYBRID_SLIDING = "hybrid_sliding"
# DeepSeek-V4: a sliding window plus compressed entries, one per `compress_ratio` tokens.
COMPRESSED_SPARSE = "compressed_sparse_attention"  # CSA: ratio 4, with lightning-indexer keys
HEAVILY_COMPRESSED = "heavily_compressed_attention"  # HCA: ratio 128, no indexer
NONE = "none"  # no cache (MLP/MoE-only block, or KV shared with another layer)

_COMPRESSED_TYPES = {COMPRESSED_SPARSE, HEAVILY_COMPRESSED}
_ATTENTION_TYPES = {GLOBAL, SLIDING, CHUNKED, HYBRID, HYBRID_SLIDING, *_COMPRESSED_TYPES}
_LOCAL_TYPES = {SLIDING, CHUNKED, HYBRID_SLIDING, *_COMPRESSED_TYPES}
_RECURRENT_TYPES = {LINEAR, HYBRID, HYBRID_SLIDING}

_LAYER_TYPE_ALIASES = {
    "full_attention": GLOBAL,
    "attention": GLOBAL,
    "global": GLOBAL,
    "global_attention": GLOBAL,
    "deepseek_sparse_attention": GLOBAL,
    "qwen_sparse_attention": GLOBAL,
    "sliding_attention": SLIDING,
    "sliding": SLIDING,
    "local": SLIDING,
    "local_attention": SLIDING,
    "chunked_attention": CHUNKED,
    "compressed_sparse_attention": COMPRESSED_SPARSE,
    "heavily_compressed_attention": HEAVILY_COMPRESSED,
    "linear_attention": LINEAR,
    "conv": LINEAR,
    "mamba": LINEAR,
    "mamba2": LINEAR,
    "ssm": LINEAR,
    "recurrent": LINEAR,
    "hybrid": HYBRID,
    "hybrid_sliding": HYBRID_SLIDING,
    "mlp": NONE,
    "moe": NONE,
    "none": NONE,
}

_NESTED_TEXT_KEYS = ("text_config", "language_config", "llm_config")


# --------------------------------------------------------------------------- loading


def _expands_v32() -> bool:
    """transformers before 5.18 caches DeepSeek-V3.2's decompressed K/V; 5.18 and later cache the latent."""
    try:
        import transformers
    except ImportError:
        return False
    try:
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) < (5, 18)


def load_config(
    model_or_config: ConfigSource | Any,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> dict[str, Any]:
    """Return a model config as a plain dict.

    ``model_or_config`` can be a dict, a transformers ``PretrainedConfig``, a path to
    a ``config.json`` file or a model directory, or a Hugging Face Hub repo id
    (only ``config.json`` is downloaded, never weights).
    """
    if isinstance(model_or_config, Mapping):
        return dict(model_or_config)
    to_dict = getattr(model_or_config, "to_dict", None)
    if callable(to_dict):  # transformers PretrainedConfig
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    path = Path(os.fspath(model_or_config))
    if path.is_dir():
        path = path / "config.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        return data
    repo_id = str(model_or_config)
    if repo_id.count("/") > 1 or repo_id.endswith(".json"):
        raise FileNotFoundError(f"No config found at {repo_id!r}")
    from huggingface_hub import hf_hub_download

    file = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision, token=token)
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config.json for {repo_id} is not a JSON object")
    return data


# --------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class LayerKV:
    """Cache geometry of one decoder layer."""

    index: int
    layer_type: str
    """One of ``full_attention``, ``sliding_attention``, ``chunked_attention``,
    ``linear_attention``, ``hybrid``, ``hybrid_sliding`` or ``none``."""
    kv_elements_per_token: int
    """K+V elements cached per token (MLA: latent + RoPE key). 0 for non-attention layers."""
    max_tokens: int | None
    """Resident-token cap for local layers (window - 1), ``None`` if the layer grows."""
    state_elements: int | None
    """Fixed recurrent-state elements (conv + SSM), ``0`` if none, ``None`` if unknown."""
    state_fp32_elements: int = 0
    """Part of ``state_elements`` held in float32 whatever the KV dtype (Gated DeltaNet's recurrent
    state, or an SSM state the config declares as float32)."""
    compress_ratio: int | None = None
    """Compressed layers (DeepSeek-V4 CSA/HCA): one compressed entry per ``compress_ratio`` tokens."""
    compressed_elements_per_entry: int = 0
    """Elements cached per compressed entry (V4 keeps one shared K=V vector)."""
    indexer_elements: int = 0
    """Lightning-indexer key elements: per compressed entry on CSA layers, per token on DSA layers."""

    @property
    def is_compressed(self) -> bool:
        return self.layer_type in _COMPRESSED_TYPES

    @property
    def has_kv(self) -> bool:
        return self.layer_type in _ATTENTION_TYPES

    @property
    def is_local(self) -> bool:
        return self.layer_type in _LOCAL_TYPES

    @property
    def is_recurrent(self) -> bool:
        return self.layer_type in _RECURRENT_TYPES

    @property
    def group(self) -> str:
        """Reporting group for the attention part: ``global``, ``sliding`` or ``recurrent``."""
        if self.is_local:
            return "sliding"
        if self.has_kv:
            return "global"
        if self.is_recurrent:
            return "recurrent"
        return "none"


@dataclass(frozen=True)
class ModelSpec:
    """Architecture facts needed for KV math, normalized from a raw config."""

    model_id: str | None
    model_type: str | None
    architecture: str
    """Family used for lineage: ``mha``, ``mqa``, ``gqa``, ``mla``, ``sliding_window``,
    ``hybrid`` or ``ssm``."""
    attention: str | None
    """Head design of the attention layers: ``mha``, ``mqa``, ``gqa`` or ``mla``."""
    num_layers: int
    num_attention_heads: int | None
    num_kv_heads: int | None
    head_dim: int | None
    kv_lora_rank: int | None
    qk_rope_head_dim: int | None
    sliding_window: int | None
    max_context_tokens: int | None
    dtype: str | None
    layers: tuple[LayerKV, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(layer.layer_type for layer in self.layers)

    @property
    def num_attention_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.has_kv)


def _get(cfg: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return value
    return None


def _int(cfg: Mapping[str, Any], *keys: str) -> int | None:
    value = _get(cfg, *keys)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _text_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the decoder (text) config, merging a few top-level fields into it."""
    for key in _NESTED_TEXT_KEYS:
        sub = cfg.get(key)
        if isinstance(sub, Mapping) and sub:
            merged = dict(sub)
            for inherited in ("torch_dtype", "dtype", "_name_or_path"):
                if merged.get(inherited) is None and cfg.get(inherited) is not None:
                    merged[inherited] = cfg[inherited]
            return merged
    return dict(cfg)


def _canonical_layer_type(value: object) -> str:
    name = str(value).strip().lower()
    return _LAYER_TYPE_ALIASES.get(name, GLOBAL if "attention" in name else name)


def _pattern_types(n_layers: int, local_if: Any) -> list[str]:
    return [SLIDING if local_if(i) else GLOBAL for i in range(n_layers)]


def _infer_layer_types(cfg: Mapping[str, Any], model_type: str, n_layers: int) -> tuple[list[str], list[str]]:
    """Per-layer cache types, following transformers' config defaults."""
    notes: list[str] = []
    explicit = _get(cfg, "layer_types", "layers_block_type")
    ratios = cfg.get("compress_ratios")
    if not explicit and isinstance(ratios, Sequence) and not isinstance(ratios, str) and ratios:
        return _compress_ratio_types(cfg, ratios, n_layers)
    if isinstance(explicit, Sequence) and not isinstance(explicit, str) and explicit:
        types = [_canonical_layer_type(t) for t in explicit]
        unknown = sorted({t for t in types if t not in _LAYER_TYPE_ALIASES.values()})
        if unknown:
            notes.append(f"Unrecognized layer types {unknown} are treated as full attention.")
            types = [GLOBAL if t in unknown else t for t in types]
        return types, notes

    sliding_window = _int(cfg, "sliding_window")
    if cfg.get("use_sliding_window") is False:
        sliding_window = None

    if model_type == "lfm2":
        idxs = cfg.get("full_attn_idxs")
        attn = set(idxs) if isinstance(idxs, Sequence) else set(range(n_layers))
        return [GLOBAL if i in attn else LINEAR for i in range(n_layers)], notes
    if model_type == "jamba":
        period = _int(cfg, "attn_layer_period") or 8
        offset = _int(cfg, "attn_layer_offset") or 4
        return [GLOBAL if i % period == offset else LINEAR for i in range(n_layers)], notes
    if model_type == "bamba":
        idxs = cfg.get("attn_layer_indices") or []
        return [GLOBAL if i in set(idxs) else LINEAR for i in range(n_layers)], notes
    if model_type == "nemotron_h" and isinstance(cfg.get("hybrid_override_pattern"), str):
        chars = {"M": LINEAR, "*": GLOBAL, "-": NONE, "E": NONE}
        return [chars.get(c, NONE) for c in cfg["hybrid_override_pattern"]], notes
    if model_type == "falcon_h1":
        return [HYBRID] * n_layers, notes
    if model_type == "qwen3_next":
        interval = _int(cfg, "full_attention_interval") or 4
        return [LINEAR if (i + 1) % interval else GLOBAL for i in range(n_layers)], notes
    if model_type == "minimax":
        return [GLOBAL if (i + 1) % 2 else LINEAR for i in range(n_layers)], notes
    if model_type in {"mamba", "mamba2", "rwkv", "rwkv5", "rwkv6", "rwkv7", "xlstm"}:
        return [LINEAR] * n_layers, notes

    # Local/global attention families.
    if model_type in {"gemma2", "gpt_oss"}:
        return _pattern_types(n_layers, lambda i: (i + 1) % 2), notes
    if model_type in {"gemma3", "gemma3_text", "cohere2"}:
        default = 6 if model_type.startswith("gemma3") else 4
        pattern = _int(cfg, "sliding_window_pattern", "_sliding_window_pattern") or default
        return _pattern_types(n_layers, lambda i: (i + 1) % pattern), notes
    if model_type in {"gemma3n", "gemma3n_text"}:
        return _pattern_types(n_layers, lambda i: (i + 1) % 5 != 0), notes
    if model_type == "olmo3":
        return _pattern_types(n_layers, lambda i: (i + 1) % 4 != 0), notes
    if model_type == "exaone4":
        if sliding_window is None:
            return [GLOBAL] * n_layers, notes
        pattern = cfg.get("sliding_window_pattern", 4)
        if isinstance(pattern, str) and pattern:
            return [SLIDING if pattern[i % len(pattern)].upper() == "L" else GLOBAL for i in range(n_layers)], notes
        period = int(pattern) if isinstance(pattern, (int, float)) and pattern else 4
        return _pattern_types(n_layers, lambda i: (i + 1) % period != 0), notes
    if model_type in {"qwen2_moe"}:
        if not cfg.get("use_sliding_window"):
            return [GLOBAL] * n_layers, notes
        mwl = _int(cfg, "max_window_layers")
        limit = n_layers if mwl is None else mwl
        return _pattern_types(n_layers, lambda i: (i + 1) % 2 and i < limit), notes
    if model_type == "smollm3":
        no_rope = cfg.get("no_rope_layers")
        if not isinstance(no_rope, Sequence) or len(no_rope) != n_layers:
            interval = _int(cfg, "no_rope_layer_interval") or 4
            no_rope = [int((i + 1) % interval != 0) for i in range(n_layers)]
        if sliding_window is None:
            return [GLOBAL] * n_layers, notes
        return [GLOBAL if has_rope else SLIDING for has_rope in no_rope], notes
    if model_type.startswith(("qwen2", "qwen3", "dots1")) and model_type not in {"qwen3_moe"}:
        if sliding_window is None:
            return [GLOBAL] * n_layers, notes
        mwl = _int(cfg, "max_window_layers")
        first_local = n_layers if mwl is None else mwl
        return _pattern_types(n_layers, lambda i: i >= first_local), notes
    if model_type in {"llama4", "llama4_text"}:
        no_rope = cfg.get("no_rope_layers")
        if not isinstance(no_rope, Sequence) or len(no_rope) != n_layers:
            interval = _int(cfg, "no_rope_layer_interval") or 4
            no_rope = [int((i + 1) % interval != 0) for i in range(n_layers)]
        if _int(cfg, "attention_chunk_size") is None:
            return [GLOBAL] * n_layers, notes
        return [CHUNKED if flag else GLOBAL for flag in no_rope], notes
    if model_type == "mistral" and "sliding_window" not in cfg:
        sliding_window = 4096  # MistralConfig default when the key is absent

    # Generic rule used by transformers' cache when no layer_types exist.
    if sliding_window is not None:
        return [SLIDING] * n_layers, notes
    if _int(cfg, "attention_chunk_size") is not None:
        return [CHUNKED] * n_layers, notes
    return [GLOBAL] * n_layers, notes


def _kv_heads(cfg: Mapping[str, Any], model_type: str, n_heads: int | None, n_layers: int) -> list[int] | None:
    if model_type == "falcon" and not cfg.get("new_decoder_architecture") and cfg.get("multi_query", True):
        return [1] * n_layers  # Falcon-7B style multi-query, whatever num_kv_heads says
    raw = _get(cfg, "num_key_value_heads", "num_kv_heads", "n_head_kv", "kv_n_heads")
    attn_config = cfg.get("attn_config")
    if raw is None and isinstance(attn_config, Mapping):
        raw = attn_config.get("kv_n_heads")  # MPT
    if raw is None and cfg.get("multi_query_attention"):  # ChatGLM
        raw = _int(cfg, "multi_query_group_num")
    if raw is None and cfg.get("multi_query") is True:  # GPT-BigCode / StarCoder
        raw = 1
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return [int(x) for x in raw]
    if raw is not None and not isinstance(raw, bool):
        return [int(raw)] * n_layers
    return [n_heads] * n_layers if n_heads else None


def _declares_fp32_state(cfg: Mapping[str, Any]) -> bool:
    declared = _get(cfg, "mamba_ssm_cache_dtype", "mamba_ssm_dtype", "ssm_state_dtype")
    return declared is not None and "32" in str(declared)


def _state_elements(cfg: Mapping[str, Any], model_type: str, hidden: int | None) -> tuple[int | None, int]:
    """Fixed recurrent-state elements per recurrent layer (conv + SSM) and the part held in float32."""
    if model_type == "lfm2":
        l_cache = _int(cfg, "conv_L_cache")
        width = _int(cfg, "conv_dim") or hidden
        return (width * l_cache if width and l_cache else None), 0
    if model_type == "qwen3_next" or _int(cfg, "linear_num_key_heads") is not None:
        # Gated DeltaNet (Qwen3-Next, Qwen3.5/3.6 ``qwen3_5*_text``, ``qwen4_exp_text``). Keyed on the
        # fields rather than the model_type, so later generations with the same layout are covered.
        # The recurrent state is computed and kept in float32 (transformers' kernels and the reference
        # path both return it in float32, and Qwen3.5 configs declare ``mamba_ssm_dtype: float32``).
        nk, dk = _int(cfg, "linear_num_key_heads"), _int(cfg, "linear_key_head_dim")
        nv, dv = _int(cfg, "linear_num_value_heads"), _int(cfg, "linear_value_head_dim")
        kernel = _int(cfg, "linear_conv_kernel_dim")
        if not (nk and dk and nv and dv and kernel):
            return None, 0
        conv = (2 * nk * dk + nv * dv) * kernel
        recurrent = nv * dk * dv
        return conv + recurrent, recurrent
    fp32 = _declares_fp32_state(cfg)
    if model_type in {"jamba", "mamba"}:
        expand = _int(cfg, "mamba_expand", "expand") or 2
        inner = _int(cfg, "intermediate_size") if model_type == "mamba" else None
        inner = inner or (expand * hidden if hidden else None)
        d_state = _int(cfg, "mamba_d_state", "state_size")
        d_conv = _int(cfg, "mamba_d_conv", "conv_kernel")
        if not (inner and d_state and d_conv):
            return None, 0
        return inner * d_conv + inner * d_state, (inner * d_state if fp32 else 0)
    # Mamba2-style families use a few naming schemes; plain Mamba2 uses unprefixed keys.
    plain = model_type == "mamba2"
    n_heads = _int(cfg, "mamba_n_heads", "mamba_num_heads", "n_mamba_heads", *(("num_heads",) if plain else ()))
    d_head = _int(cfg, "mamba_d_head", "mamba_head_dim", "mamba_headdim", *(("head_dim",) if plain else ()))
    d_state = _int(cfg, "mamba_d_state", "ssm_state_size", "state_size")
    d_conv = _int(cfg, "mamba_d_conv", "conv_kernel")
    groups = _int(cfg, "mamba_n_groups", "mamba_ngroups", "n_groups") or 1
    expand = _int(cfg, "mamba_expand", "expand") or 2
    inner = _int(cfg, "mamba_d_ssm") or (n_heads * d_head if n_heads and d_head else None)
    inner = inner or (expand * hidden if hidden else None)
    if not (n_heads and d_head and d_state and d_conv and inner):
        return None, 0
    conv = (inner + 2 * groups * d_state) * d_conv
    ssm = n_heads * d_head * d_state
    return conv + ssm, (ssm if fp32 else 0)


def _compress_ratio_types(cfg: Mapping[str, Any], ratios: Sequence[Any], n_layers: int) -> tuple[list[str], list[str]]:
    """DeepSeek-V4 ``compress_ratios`` (0 = sliding only, 4 = CSA, 128 = HCA) as layer types."""
    rates = _compress_rates(cfg)
    csa, hca = rates[COMPRESSED_SPARSE], rates[HEAVILY_COMPRESSED]
    known = {0: SLIDING, csa: COMPRESSED_SPARSE, hca: HEAVILY_COMPRESSED}
    notes: list[str] = []
    types = []
    for ratio in list(ratios)[:n_layers]:
        kind = known.get(int(ratio))
        if kind is None:
            kind = GLOBAL
            notes.append(f"compress ratio {ratio} is not modelled yet; that layer is predicted as full attention.")
        types.append(kind)
    return types, sorted(set(notes))


def _compress_rates(cfg: Mapping[str, Any]) -> dict[str, int]:
    raw = cfg.get("compress_rates")
    rates: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return {
        COMPRESSED_SPARSE: int(rates.get(COMPRESSED_SPARSE) or _int(cfg, "compress_rate_csa") or 4),
        HEAVILY_COMPRESSED: int(rates.get(HEAVILY_COMPRESSED) or _int(cfg, "compress_rate_hca") or 128),
    }


def _dsa_indexer_elements(cfg: Mapping[str, Any]) -> int:
    """Per-token lightning-indexer key elements of DeepSeek Sparse Attention layers (0 when absent)."""
    heads = _int(cfg, "indexer_kv_heads")
    head_dim = _int(cfg, "indexer_head_dim")
    if heads and head_dim:
        return heads * head_dim
    return _int(cfg, "index_head_dim") or 0


def spec_from_config(config: Mapping[str, Any], model_id: str | None = None) -> ModelSpec:
    """Normalize a raw ``config.json`` dict into a :class:`ModelSpec`."""
    top = dict(config)
    cfg = _text_config(top)
    model_type = str(cfg.get("model_type") or top.get("model_type") or "").lower()
    notes: list[str] = []

    explicit_types = _get(cfg, "layer_types", "layers_block_type")
    n_layers = _int(cfg, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    if isinstance(explicit_types, Sequence) and not isinstance(explicit_types, str) and explicit_types:
        n_layers = len(explicit_types) if n_layers is None or model_type == "nemotron_h" else n_layers
    if model_type == "nemotron_h" and isinstance(cfg.get("hybrid_override_pattern"), str):
        n_layers = len(cfg["hybrid_override_pattern"])
    if not n_layers:
        raise ValueError("Config has no layer count (num_hidden_layers / n_layer / num_layers)")

    n_heads = _int(cfg, "num_attention_heads", "n_head", "num_heads", "n_heads")
    hidden = _int(cfg, "hidden_size", "n_embd", "d_model", "dim", "model_dim")
    head_dim = _int(cfg, "head_dim", "kv_channels", "attention_head_dim", "d_head")
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads
    kv_heads = _kv_heads(cfg, model_type, n_heads, n_layers)

    kv_lora_rank = _int(cfg, "kv_lora_rank")
    rope_dim = _int(cfg, "qk_rope_head_dim")
    is_mla = kv_lora_rank is not None
    if is_mla and rope_dim is None:
        rope_dim = 0
        notes.append("MLA config has no qk_rope_head_dim; assuming 0.")

    layer_types, type_notes = _infer_layer_types(cfg, model_type, n_layers)
    notes.extend(type_notes)
    if len(layer_types) != n_layers:
        layer_types = (layer_types + [GLOBAL] * n_layers)[:n_layers]

    shared = _int(cfg, "num_kv_shared_layers")
    if shared:
        layer_types = layer_types[: n_layers - shared] + [NONE] * shared
        notes.append(f"The last {shared} layers reuse KV from earlier layers and add none of their own.")

    sliding_window = _int(cfg, "sliding_window")
    if cfg.get("use_sliding_window") is False and not any(t in _LOCAL_TYPES for t in layer_types):
        sliding_window = None
    if model_type == "mistral" and "sliding_window" not in cfg:
        sliding_window = 4096
    chunk = _int(cfg, "attention_chunk_size")

    state, state_fp32 = _state_elements(cfg, model_type, hidden)
    rates = _compress_rates(cfg)
    # DeepSeek-V4 keeps one shared K=V vector per token (``values is keys``), not separate K and V.
    shared_kv = model_type.startswith("deepseek_v4") or any(t in _COMPRESSED_TYPES for t in layer_types)
    dsa_indexer = 0 if shared_kv else _dsa_indexer_elements(cfg)
    csa_indexer = _int(cfg, "index_head_dim") or 0
    layers: list[LayerKV] = []
    for i, layer_type in enumerate(layer_types):
        ratio: int | None = None
        per_entry = 0
        indexer = 0
        if layer_type in _ATTENTION_TYPES:
            if is_mla:
                elems = int(kv_lora_rank or 0) + int(rope_dim or 0)
            else:
                if kv_heads is None or head_dim is None:
                    raise ValueError("Config lacks attention head counts or head_dim")
                elems = (1 if shared_kv else 2) * kv_heads[i] * head_dim
            indexer = dsa_indexer
        else:
            elems = 0
        max_tokens: int | None = None
        if layer_type in {SLIDING, HYBRID_SLIDING} or layer_type in _COMPRESSED_TYPES:
            if sliding_window is None:
                raise ValueError(f"Layer {i} is sliding-window but the config has no sliding_window")
            max_tokens = max(sliding_window - 1, 0)
        elif layer_type == CHUNKED:
            window = chunk or sliding_window
            if window is None:
                raise ValueError(f"Layer {i} is chunked but the config has no attention_chunk_size")
            max_tokens = max(window - 1, 0)
        if layer_type in _COMPRESSED_TYPES:
            ratio = rates[layer_type]
            per_entry = elems
            indexer = csa_indexer if layer_type == COMPRESSED_SPARSE else 0
        recurrent = layer_type in _RECURRENT_TYPES
        layer_state: int | None = state if recurrent else 0
        layers.append(
            LayerKV(
                i,
                layer_type,
                elems,
                max_tokens,
                layer_state,
                state_fp32_elements=state_fp32 if recurrent and state is not None else 0,
                compress_ratio=ratio,
                compressed_elements_per_entry=per_entry,
                indexer_elements=indexer,
            )
        )

    has_kv = any(layer.has_kv for layer in layers)
    has_rec = any(layer.is_recurrent for layer in layers)
    has_local = any(layer.is_local for layer in layers)
    has_compressed = any(layer.is_compressed for layer in layers)
    has_indexer = any(layer.indexer_elements and not layer.is_compressed for layer in layers)

    attention: str | None
    if not has_kv:
        attention = None
    elif is_mla:
        attention = "mla"
    else:
        heads_used = {kv_heads[i] for i, layer in enumerate(layers) if layer.has_kv} if kv_heads else set()
        if n_heads and heads_used == {n_heads}:
            attention = "mha"
        elif heads_used == {1} and (n_heads or 0) > 1:
            attention = "mqa"
        else:
            attention = "gqa"

    if has_rec and has_kv:
        architecture = "hybrid"
    elif has_rec:
        architecture = "ssm"
    elif has_compressed:
        architecture = "compressed"
    elif has_indexer:
        architecture = "dsa"
    elif is_mla:
        architecture = "mla"
    elif has_local:
        architecture = "sliding_window"
    else:
        architecture = attention or "mha"

    if has_rec and state is None:
        notes.append(f"Recurrent-state size is not known for model_type {model_type!r}; reported as null.")
    if has_rec and state is not None:
        if state_fp32:
            notes.append(
                "Recurrent-state bytes: the SSM/DeltaNet state is sized in float32 (transformers computes and keeps "
                "it in float32, and the config declares it so); the convolution state is sized at the KV dtype."
            )
        else:
            notes.append("Recurrent-state bytes are sized at the KV dtype, as transformers allocates them.")
    if has_compressed:
        csa = sum(1 for layer in layers if layer.layer_type == COMPRESSED_SPARSE)
        hca = sum(1 for layer in layers if layer.layer_type == HEAVILY_COMPRESSED)
        notes.append(
            f"Compressed attention: every layer keeps a {sliding_window}-token window of one shared K=V vector; "
            f"{csa} CSA layers add one compressed entry per {rates[COMPRESSED_SPARSE]} tokens plus "
            f"{csa_indexer}-dim lightning-indexer keys, {hca} HCA layers one entry per "
            f"{rates[HEAVILY_COMPRESSED]} tokens. Tokens waiting to fill a window (at most ratio - 1 per layer) "
            "are held in engine buffers (measured as `pending`) and are not part of the architectural KV; "
            "transformers also keeps separate K and V tensors on the sliding-only layers."
        )
    if has_indexer:
        notes.append(
            f"DeepSeek Sparse Attention: every attention layer also caches a {dsa_indexer}-dim lightning-indexer "
            "key per token (reported as its own `indexer` group)."
        )
        if model_type == "deepseek_v32" and _expands_v32():
            notes.append(
                "transformers before 5.18 caches DeepSeek-V3.2's decompressed per-head K/V (DynamicIndexedLayer), "
                "not the latent: measured bytes exceed the architectural KV by that engine layout."
            )
    if is_mla and n_heads:
        nope = _int(cfg, "qk_nope_head_dim") or 0
        v_dim = _int(cfg, "v_head_dim") or 0
        expanded = n_heads * (nope + int(rope_dim or 0)) + n_heads * v_dim
        if expanded:
            notes.append(
                "Architectural MLA KV is the compressed latent + RoPE key "
                f"({int(kv_lora_rank or 0) + int(rope_dim or 0)} elements/token/layer). transformers 5.x caches "
                "the latent for DeepSeek-V2/V3; engines that cache decompressed per-head K/V hold "
                f"{expanded} elements/token/layer."
            )
    if has_local:
        caps = sorted({layer.max_tokens for layer in layers if layer.max_tokens is not None})
        local = sum(1 for layer in layers if layer.is_local)
        notes.append(
            f"{local} of {n_layers} layers are local; each keeps at most {', '.join(map(str, caps))} "
            "resident tokens (window - 1: the current token attends to itself plus window - 1 cached ones)."
        )

    max_ctx = _int(cfg, "max_position_embeddings", "n_positions", "max_sequence_length", "seq_length", "n_ctx")
    dtype = _get(cfg, "torch_dtype", "dtype")
    model_name = model_id or _get(top, "_name_or_path", "name_or_path") or None

    return ModelSpec(
        model_id=str(model_name) if model_name else None,
        model_type=model_type or None,
        architecture=architecture,
        attention=attention,
        num_layers=n_layers,
        num_attention_heads=n_heads,
        num_kv_heads=(max(kv_heads) if kv_heads else None),
        head_dim=None if is_mla else head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim if is_mla else None,
        sliding_window=sliding_window if has_local else None,
        max_context_tokens=max_ctx,
        dtype=str(dtype) if dtype is not None else None,
        layers=tuple(layers),
        notes=tuple(notes),
    )
