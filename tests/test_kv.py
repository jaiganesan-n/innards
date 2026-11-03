"""KV calculator unit tests. No network: configs come from fixtures copied from the Hub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CONFIGS, load_fixture_config
from innards.kv import (
    DTYPE_BYTES,
    ModelSpec,
    dtype_bytes,
    load_config,
    normalize_dtype,
    predict,
    spec_from_config,
)

DTYPES = ["float32", "float16", "bfloat16", "float8"]


# --------------------------------------------------------------------- acceptance


@pytest.mark.parametrize("dtype", DTYPES)
def test_gpt2_bytes_per_token(dtype: str) -> None:
    p = predict(CONFIGS / "gpt2.json", 100, dtype)
    b = DTYPE_BYTES[dtype]
    assert p.bytes_per_token == 2 * 12 * 12 * 64 * b
    assert p.kv_bytes == 100 * 2 * 12 * 12 * 64 * b
    assert p.architecture == "mha"


@pytest.mark.parametrize("dtype", DTYPES)
def test_qwen3_0_6b_bytes_per_token(dtype: str) -> None:
    p = predict(CONFIGS / "qwen3-0.6b.json", 1, dtype)
    assert p.bytes_per_token == 2 * 28 * 8 * 128 * DTYPE_BYTES[dtype]
    assert p.architecture == "gqa"


def test_llama_3_2_3b_fp16() -> None:
    assert predict(CONFIGS / "llama-3.2-3b.json", 1, "fp16").bytes_per_token == 114_688


def test_phi_3_mini_fp16() -> None:
    p = predict(CONFIGS / "phi-3-mini-4k-instruct.json", 1, "fp16")
    assert p.bytes_per_token == 393_216
    assert p.attention == "mha"


def test_deepseek_v2_lite_fp16() -> None:
    p = predict(CONFIGS / "deepseek-v2-lite.json", 1, "fp16")
    assert p.bytes_per_token == 31_104
    assert p.architecture == "mla"
    assert "formula" not in p.notes
    assert p.formula.startswith("27 layers × (512 latent + 64 RoPE) × 2 B = 31,104")


def test_design_doc_example_is_consistent() -> None:
    # 3,304 tokens of Llama-3.2-3B in fp16 = 378,929,152 bytes (design doc TurnRecord example).
    assert predict(CONFIGS / "llama-3.2-3b.json", 3304, "float16").kv_bytes == 378_929_152


# --------------------------------------------------------------------- dtypes


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("fp32", "float32"),
        ("FP16", "float16"),
        ("half", "float16"),
        ("bf16", "bfloat16"),
        ("torch.bfloat16", "bfloat16"),
        ("fp8", "float8"),
        ("float8_e4m3fn", "float8"),
        ("fp8_e5m2", "float8"),
    ],
)
def test_dtype_aliases(alias: str, canonical: str) -> None:
    assert normalize_dtype(alias) == canonical
    assert dtype_bytes(alias) == DTYPE_BYTES[canonical]


def test_unknown_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported KV dtype"):
        predict(CONFIGS / "gpt2.json", 1, "int3")


# --------------------------------------------------------------------- config handling


def test_head_dim_derived_when_absent() -> None:
    spec = spec_from_config(load_fixture_config("gpt2"))
    assert spec.head_dim == 768 // 12
    assert spec.num_layers == 12
    assert spec.num_kv_heads == 12


def test_explicit_head_dim_wins_over_hidden_over_heads() -> None:
    spec = spec_from_config(load_fixture_config("qwen3-0.6b"))
    assert spec.head_dim == 128  # hidden_size / heads would give 64


def test_legacy_chatglm_keys() -> None:
    cfg = {
        "model_type": "chatglm",
        "num_layers": 28,
        "num_attention_heads": 32,
        "hidden_size": 4096,
        "kv_channels": 128,
        "multi_query_attention": True,
        "multi_query_group_num": 2,
    }
    p = predict(cfg, 10, "fp16")
    assert p.bytes_per_token == 2 * 28 * 2 * 128 * 2
    assert p.attention == "gqa"


def test_multi_query_flag_is_mqa() -> None:
    cfg = {"model_type": "gpt_bigcode", "n_layer": 4, "n_head": 8, "n_embd": 512, "multi_query": True}
    p = predict(cfg, 1, "fp16")
    assert p.attention == "mqa"
    assert p.bytes_per_token == 2 * 4 * 1 * 64 * 2


def test_falcon_legacy_multi_query() -> None:
    cfg = {"model_type": "falcon", "num_hidden_layers": 32, "num_attention_heads": 71, "hidden_size": 4544}
    assert predict(cfg, 1, "bf16").attention == "mqa"


def test_nested_text_config_is_used() -> None:
    text = load_fixture_config("gemma-3-1b-it")
    wrapped = {"model_type": "gemma3", "text_config": text, "vision_config": {"hidden_size": 1152}}
    assert predict(wrapped, 4096).kv_bytes == predict(text, 4096).kv_bytes


def test_load_config_from_dir_and_dict(tmp_path: Path) -> None:
    cfg = load_fixture_config("gpt2")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert load_config(tmp_path) == cfg
    assert load_config(tmp_path / "config.json") == cfg
    assert load_config(cfg) == cfg


def test_missing_local_path_is_not_sent_to_hub(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope" / "config.json"))


def test_spec_passthrough() -> None:
    spec = spec_from_config(load_fixture_config("gpt2"))
    assert isinstance(spec, ModelSpec)
    assert predict(spec, 10).kv_bytes == predict(CONFIGS / "gpt2.json", 10).kv_bytes


def test_context_beyond_max_is_noted() -> None:
    p = predict(CONFIGS / "gpt2.json", 4096)
    assert any("exceeds" in note for note in p.notes)


# --------------------------------------------------------------------- sliding window


def test_gemma3_layer_pattern_and_window_cap() -> None:
    p = predict(CONFIGS / "gemma-3-1b-it.json", 10_000, "bf16")
    types = [layer.layer_type for layer in p.layers]
    global_layers = [i for i, t in enumerate(types) if t == "full_attention"]
    assert global_layers == [5, 11, 17, 23]
    per_token = 2 * 1 * 256 * 2
    assert p.by_layer_type["sliding"] == 22 * 511 * per_token  # window 512 -> 511 resident
    assert p.by_layer_type["global"] == 4 * 10_000 * per_token
    assert p.kv_bytes == sum(p.per_layer_bytes)
    assert p.architecture == "sliding_window"


def test_local_layers_stop_growing_past_window() -> None:
    before = predict(CONFIGS / "gemma-3-1b-it.json", 511)
    after = predict(CONFIGS / "gemma-3-1b-it.json", 5_000)
    for layer, a, b in zip(before.layers, before.per_layer_bytes, after.per_layer_bytes, strict=True):
        if layer.is_local:
            assert a == b
        else:
            assert b > a


def test_explicit_layer_types_gpt_oss() -> None:
    p = predict(CONFIGS / "gpt-oss-20b.json", 1000)
    assert sum(1 for layer in p.layers if layer.is_local) == 12
    assert {layer.max_tokens for layer in p.layers if layer.is_local} == {127}
    assert p.bytes_per_token_global == p.bytes_per_token // 2


def test_phi3_config_window_applies_to_every_layer() -> None:
    p = predict(CONFIGS / "phi-3-mini-4k-instruct.json", 4096, "fp16")
    assert all(layer.is_local and layer.max_tokens == 2046 for layer in p.layers)
    assert p.kv_bytes == 2046 * 393_216


def test_qwen_sliding_flag_off_means_global() -> None:
    cfg = load_fixture_config("qwen3-0.6b") | {"sliding_window": 4096, "use_sliding_window": False}
    assert predict(cfg, 1).architecture == "gqa"


def test_qwen_max_window_layers() -> None:
    cfg = load_fixture_config("qwen3-0.6b") | {
        "sliding_window": 1024,
        "use_sliding_window": True,
        "max_window_layers": 20,
    }
    p = predict(cfg, 1)
    assert [layer.is_local for layer in p.layers] == [False] * 20 + [True] * 8


def test_chunked_attention_llama4_style() -> None:
    cfg = {
        "model_type": "llama4_text",
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "attention_chunk_size": 16,
    }
    p = predict(cfg, 100)
    assert [layer.layer_type for layer in p.layers].count("chunked_attention") == 6
    assert {layer.max_tokens for layer in p.layers if layer.is_local} == {15}


# --------------------------------------------------------------------- MLA


def test_mla_notes_engine_expansion() -> None:
    p = predict(CONFIGS / "deepseek-v2-lite.json", 1)
    assert any("5120 elements" in note for note in p.notes)


# --------------------------------------------------------------------- hybrid


def test_granite_hybrid_only_attention_layers_grow() -> None:
    small = predict(CONFIGS / "granite-4.0-h-tiny.json", 100, "bf16")
    large = predict(CONFIGS / "granite-4.0-h-tiny.json", 10_000, "bf16")
    assert small.architecture == "hybrid"
    assert sum(1 for layer in small.layers if layer.has_kv) == 4
    for layer, a, b in zip(small.layers, small.per_layer_bytes, large.per_layer_bytes, strict=True):
        if layer.is_recurrent:
            assert a == b > 0
        else:
            assert b > a
    conv = (2 * 1536 + 2 * 1 * 128) * 4
    ssm = 48 * 64 * 128
    assert small.recurrent_state_bytes == 36 * (conv + ssm) * 2
    assert small.by_layer_type["recurrent"] == small.recurrent_state_bytes
    assert small.bytes_per_token == 2 * 4 * 4 * 128 * 2
    assert small.total_bytes == small.kv_bytes + small.recurrent_state_bytes


def test_lfm2_hybrid() -> None:
    p = predict(CONFIGS / "lfm2-350m.json", 10, "bf16")
    assert [i for i, layer in enumerate(p.layers) if layer.has_kv] == [2, 5, 8, 10, 12, 14]
    assert p.recurrent_state_bytes == 10 * 1024 * 3 * 2


def test_qwen3_next_hybrid() -> None:
    p = predict(CONFIGS / "qwen3-next-80b-a3b-instruct.json", 10, "bf16")
    assert [i for i, layer in enumerate(p.layers) if layer.has_kv] == list(range(3, 48, 4))
    conv = (2 * 16 * 128 + 32 * 128) * 4
    # the convolution state is at the KV dtype; the DeltaNet state is kept in float32
    assert p.recurrent_state_bytes == 36 * (conv * 2 + 32 * 128 * 128 * 4)


def test_gated_deltanet_state_is_keyed_on_fields() -> None:
    """Qwen3.5/3.6 (qwen3_5_text) keep Qwen3-Next's Gated DeltaNet layout under a new model_type."""
    cfg = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "hidden_size": 1024,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
    }
    p = predict(cfg, 10, "bf16")
    conv = (2 * 16 * 128 + 16 * 128) * 4
    assert p.recurrent_state_bytes == 3 * (conv * 2 + 16 * 128 * 128 * 4)
    assert p.total_bytes == p.kv_bytes + p.recurrent_state_bytes
    assert any("sized in float32" in note for note in p.notes)


def test_unknown_recurrent_state_is_null() -> None:
    cfg = {
        "model_type": "made_up_hybrid",
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "hidden_size": 256,
        "layer_types": ["mamba", "attention", "mamba", "attention"],
    }
    p = predict(cfg, 10)
    assert p.recurrent_state_bytes is None
    assert p.total_bytes == p.kv_bytes
    assert any("not known" in note for note in p.notes)


def test_pure_ssm() -> None:
    cfg = {
        "model_type": "mamba2",
        "num_hidden_layers": 2,
        "hidden_size": 256,
        "num_heads": 8,
        "head_dim": 64,
        "state_size": 16,
        "n_groups": 1,
        "conv_kernel": 4,
        "expand": 2,
    }
    p = predict(cfg, 10_000)
    assert p.architecture == "ssm"
    assert p.kv_bytes == 0
    assert p.recurrent_state_bytes == 2 * ((512 + 32) * 4 + 8 * 64 * 16) * 2


# --------------------------------------------------------------------- capacity, batch


def test_capacity_from_memory_budget() -> None:
    budget = 4 * 2**30
    p = predict(CONFIGS / "llama-3.2-3b.json", 0, "fp16", memory_bytes=budget)
    assert p.capacity_tokens == budget // 114_688
    assert predict(CONFIGS / "llama-3.2-3b.json", p.capacity_tokens or 0).kv_bytes <= budget


def test_capacity_capped_by_max_context_for_sliding() -> None:
    p = predict(CONFIGS / "phi-3-mini-4k-instruct.json", 0, memory_bytes=2**40)
    assert p.capacity_tokens == 4096


def test_batch_size_scales() -> None:
    one = predict(CONFIGS / "gpt2.json", 50)
    four = predict(CONFIGS / "gpt2.json", 50, batch_size=4)
    assert four.kv_bytes == 4 * one.kv_bytes


def test_zero_context() -> None:
    p = predict(CONFIGS / "granite-4.0-h-tiny.json", 0)
    assert p.total_bytes == 0


def test_to_dict_is_json_serializable() -> None:
    json.dumps(predict(CONFIGS / "gemma-3-1b-it.json", 100).to_dict())


# --------------------------------------------------------------------- transformers parity


@pytest.mark.hf
@pytest.mark.parametrize(
    "name", ["gpt2", "qwen3-0.6b", "gemma-3-1b-it", "gpt-oss-20b", "granite-4.0-h-tiny", "lfm2-350m"]
)
def test_layer_types_match_transformers(name: str) -> None:
    """Our pure-Python layer pattern must equal what transformers builds its cache from."""
    from transformers import AutoConfig
    from transformers.cache_utils import get_layer_types_and_kwargs

    raw = load_fixture_config(name)
    hf_config = AutoConfig.for_model(**raw).get_text_config(decoder=True)
    hf_types, _ = get_layer_types_and_kwargs(hf_config)
    remap = {"conv": "linear_attention", "mamba": "linear_attention", "attention": "full_attention"}
    ours = [layer.layer_type for layer in spec_from_config(raw).layers]
    assert ours == [remap.get(t, t) for t in hf_types]


# --------------------------------------------------------------------- live verification


@pytest.mark.network
@pytest.mark.parametrize("fixture", sorted(json.loads((CONFIGS / "sources.json").read_text())))
def test_fixture_matches_hub(fixture: str) -> None:
    """Fixtures were copied from the Hub; re-check the fields the calculator reads."""
    repo = json.loads((CONFIGS / "sources.json").read_text())[fixture]
    live = load_config(repo)
    local = json.loads((CONFIGS / fixture).read_text())
    keys = [
        "model_type",
        "num_hidden_layers",
        "n_layer",
        "num_attention_heads",
        "n_head",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "n_embd",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "sliding_window",
        "sliding_window_pattern",
        "layer_types",
        "full_attn_idxs",
        "use_sliding_window",
        "compress_ratios",
        "index_head_dim",
        "hc_mult",
        "hybrid_override_pattern",
        "moe_latent_size",
        "mamba_ssm_cache_dtype",
        "n_routed_experts",
        "num_experts_per_tok",
    ]
    for key in keys:
        assert live.get(key) == local.get(key), f"{fixture}: {key}"


# --------------------------------------------------------------------- more transformers parity


@pytest.mark.hf
@pytest.mark.parametrize(
    ("model_type", "extra"),
    [
        ("qwen3_moe", {"use_sliding_window": True, "sliding_window": 64, "max_window_layers": 2}),
        ("qwen2_moe", {"use_sliding_window": True, "sliding_window": 64, "max_window_layers": 3}),
        ("qwen2", {"use_sliding_window": True, "sliding_window": 64, "max_window_layers": 0}),
        ("qwen3", {"use_sliding_window": True, "sliding_window": 64, "max_window_layers": 2}),
        ("smollm3", {"use_sliding_window": True, "sliding_window": 64}),
        ("falcon", {"multi_query": True}),
        ("falcon", {"new_decoder_architecture": True, "num_kv_heads": 2}),
        ("gpt_bigcode", {"multi_query": True}),
    ],
)
def test_layer_types_and_heads_match_transformers(model_type: str, extra: dict[str, object]) -> None:
    from transformers import AutoConfig
    from transformers.cache_utils import get_layer_types_and_kwargs

    base = {"num_hidden_layers": 6, "num_attention_heads": 4, "hidden_size": 64}
    if model_type == "gpt_bigcode":
        base = {"n_layer": 6, "n_head": 4, "n_embd": 64}
    hf_config = AutoConfig.for_model(model_type, **base, **extra)
    hf_types, _ = get_layer_types_and_kwargs(hf_config.get_text_config(decoder=True))
    for source in (hf_config.to_dict(), {"model_type": model_type, **base, **extra}):
        spec = spec_from_config(source)
        assert [layer.layer_type for layer in spec.layers] == list(hf_types), source
    if model_type == "falcon":
        expected = 2 if extra.get("new_decoder_architecture") else 1
        assert spec_from_config(hf_config.to_dict()).num_kv_heads == expected


# --------------------------------------------------------------------- frontier lineup (config only)


def test_deepseek_v4_flash_compressed_attention() -> None:
    """CSA/HCA: a 128-token window on every layer plus pooled entries (lab finding 1)."""
    p = predict(CONFIGS / "deepseek-v4-flash.json", 131072, "bf16")
    assert p.architecture == "compressed"
    kinds = [layer.layer_type for layer in p.layers]
    assert kinds.count("compressed_sparse_attention") == 21
    assert kinds.count("heavily_compressed_attention") == 20
    assert kinds.count("sliding_attention") == 2
    window = 43 * 127 * 512 * 2
    csa = 21 * (131072 // 4) * (512 + 128) * 2
    hca = 20 * (131072 // 128) * 512 * 2
    assert p.total_bytes == window + csa + hca
    assert round(p.total_bytes / 2**20) == 865  # the V4 reference estimate at 128K tokens
    assert p.by_layer_type["indexer"] == 21 * (131072 // 4) * 128 * 2
    assert p.by_layer_type["sliding"] == window


def test_deepseek_v32_indexer_keys() -> None:
    """DSA: every MLA layer also caches a lightning-indexer key per token (lab finding 4)."""
    p = predict(CONFIGS / "deepseek-v3.2.json", 1000, "bf16")
    assert p.architecture == "dsa"
    assert p.attention == "mla"
    assert p.by_layer_type["global"] == 61 * 1000 * (512 + 64) * 2
    assert p.by_layer_type["indexer"] == 61 * 1000 * 128 * 2
    assert p.bytes_per_token == 61 * (512 + 64 + 128) * 2


def test_nemotron_3_ssm_state_is_float32() -> None:
    """``mamba_ssm_cache_dtype: float32``: the SSM part is sized in float32 (lab finding 6)."""
    p = predict(CONFIGS / "nemotron-3-nano-30b-a3b.json", 4096, "bf16")
    mamba = [layer for layer in p.layers if layer.is_recurrent]
    assert len(mamba) == 23
    inner = 64 * 64
    conv = (inner + 2 * 8 * 128) * 4
    ssm = 64 * 64 * 128
    assert p.recurrent_state_bytes == 23 * (conv * 2 + ssm * 4)
    assert sum(1 for layer in p.layers if layer.has_kv) == 6
