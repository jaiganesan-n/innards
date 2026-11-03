"""observe on small original checkpoints (downloaded once into the Hugging Face cache; needs the Hub).

Each architecture Innards measures has a small checkpoint here: gpt2 (MHA, no chat template),
Qwen3-0.6B (GQA), Gemma 3 270M (sliding window), LFM2-350M (short-convolution hybrid) and
Qwen3.5-0.8B (Gated DeltaNet hybrid). Runs on CPU in float32. Replies must be identical with and
without the hooks, and the KV predicted from the config must equal the cache transformers holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.hf, pytest.mark.network, pytest.mark.slow]

MESSAGES = ["Remember this: my name is Arun.", "What is my name?", "Name one way to make a KV cache smaller."]
# A long first message: past Gemma 3's 512-token sliding window in one turn.
LONG = " ".join(
    f"Point {i}: a language model keeps the keys and values of every earlier token so it does not recompute them."
    for i in range(1, 31)
)
GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}


def load(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    return model, tok


def chat(model: Any, tok: Any, messages: list[str], watch: Any = None, new_tokens: int = 8) -> list[list[int]]:
    """Your code: a growing conversation, one ``generate`` per message (plain text when there is no template)."""
    history: list[dict[str, str]] = []
    outputs = []
    for message in messages:
        history.append({"role": "user", "content": message})
        if tok.chat_template:
            text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            text = "\n".join(m["content"] for m in history) + "\n"
        inputs = tok(text, return_tensors="pt", add_special_tokens=False)
        if watch is not None:
            watch.next_turn(message)
        out = model.generate(**inputs, max_new_tokens=new_tokens, min_new_tokens=new_tokens, **GREEDY)
        reply = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        history.append({"role": "assistant", "content": reply})
        outputs.append(out[0].tolist())
    return outputs


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"])
def test_replies_identical_with_and_without_hooks(model_id: str, tmp_path: Path) -> None:
    import innards

    model, tok = load(model_id)
    plain = chat(model, tok, MESSAGES[:2])
    watch = innards.observe(model, tok, model_id=model_id, run="pair", root=str(tmp_path))
    try:
        observed = chat(model, tok, MESSAGES[:2], watch)
        watch.wait()
        for record in watch.records:
            assert record.kv.measured_bytes == record.kv.predicted_bytes
        if model_id.endswith("3.5-0.8B"):
            assert {"gated_deltanet", "gated_attention"} <= set(watch.records[-1].features)
    finally:
        watch.close()
    assert observed == plain


@pytest.mark.parametrize(("model_id", "architecture"), [("openai-community/gpt2", "mha"), ("Qwen/Qwen3-0.6B", "gqa")])
def test_measured_equals_predicted_over_three_turns(model_id: str, architecture: str) -> None:
    import innards
    from innards.kv import dtype_bytes, predict

    model, tok = load(model_id)
    with innards.observe(model, tok, store=False) as watch:
        chat(model, tok, MESSAGES, watch, new_tokens=6)
        watch.wait()
        records = list(watch.records)
        spec, kv_dtype = watch.spec, watch.kv_dtype
    per_token = sum(layer.kv_elements_per_token for layer in spec.layers) * dtype_bytes(kv_dtype)
    assert [r.turn for r in records] == [1, 2, 3]
    for r in records:
        assert r.model.architecture == architecture
        assert r.kv.measured_source == "hf_cache"
        assert r.kv.measured_bytes == r.kv.predicted_bytes
        assert r.kv.cache_tokens == r.tokens.context_after - 1
        assert r.kv.measured_bytes == r.kv.cache_tokens * per_token
        assert tuple(r.kv.per_layer_bytes) == predict(spec, r.kv.cache_tokens, kv_dtype).per_layer_bytes
        assert r.tokens.generated == 6
        assert r.timing.ttft_ms is not None
        assert r.timing.ttft_ms > 0
        assert r.memory.weights_bytes
        assert r.memory.source == "psutil_rss"
    assert records[0].kv.growth_bytes == records[0].kv.measured_bytes
    assert records[0].model.lineage is not None


def test_sliding_window_layers_stop_growing() -> None:
    import innards

    model, tok = load("unsloth/gemma-3-270m-it")
    with innards.observe(model, tok, store=False) as watch:
        chat(model, tok, [LONG, *MESSAGES[:2]], watch, new_tokens=6)
        watch.wait()
        records, spec = list(watch.records), watch.spec
    local = [i for i, layer in enumerate(spec.layers) if layer.is_local]
    full = [i for i, layer in enumerate(spec.layers) if layer.has_kv and not layer.is_local]
    assert local
    assert full
    first, last = records[0], records[-1]
    assert first.kv.cache_tokens is not None
    assert first.kv.cache_tokens > spec.sliding_window  # already past the window
    for r in records:
        assert r.kv.measured_bytes == r.kv.predicted_bytes
        sliding = {r.kv.per_layer_bytes[i] for i in local}
        assert sliding == {first.kv.per_layer_bytes[local[0]]}  # every sliding layer holds one full window
        assert r.kv.per_layer_bytes[full[0]] > r.kv.per_layer_bytes[local[0]]
    assert last.kv.per_layer_bytes[full[0]] > first.kv.per_layer_bytes[full[0]]
    assert first.model.architecture == "sliding_window"


@pytest.mark.parametrize("model_id", ["LiquidAI/LFM2-350M", "Qwen/Qwen3.5-0.8B"])
def test_hybrid_reports_recurrent_state_separately(model_id: str) -> None:
    import innards

    model, tok = load(model_id)
    with innards.observe(model, tok, features="light", store=False) as watch:
        chat(model, tok, MESSAGES, watch, new_tokens=6)
        watch.wait()
        records = list(watch.records)
    recurrent = [r.kv.by_layer_type["recurrent"] for r in records]
    assert recurrent[0] > 0
    assert len(set(recurrent)) == 1  # fixed-size state
    assert records[-1].kv.by_layer_type["global"] > records[0].kv.by_layer_type["global"]
    for r in records:
        assert r.kv.by_layer_type["global"] == r.kv.predicted_by_layer_type["global"]
        assert r.kv.measured_bytes == r.kv.predicted_bytes
    assert records[0].model.architecture == "hybrid"
