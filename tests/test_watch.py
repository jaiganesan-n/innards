"""innards.observe on tiny random-weight replicas of real architectures (offline).

Replicas exercise the exact transformers module classes each probe hooks; their numbers mean nothing.
The same checks on original checkpoints are in test_watch_models.py (needs the Hub).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from innards.kv.config import _expands_v32
from replicas import chat_ids, replica, tokenizer

pytestmark = pytest.mark.hf

MESSAGES = [
    "remember my name is priya and my favourite colour is green",
    "what is my name",
    "tell me a story about the city",
]


def converse(model: Any, turns: int = 3, *, reuse_cache: bool = False, watch: Any = None, new: int = 6) -> list[Any]:
    """A chat loop like a runner's: render the history, generate, append the reply."""
    from transformers import DynamicCache

    tok = tokenizer()
    history: list[dict[str, str]] = []
    cache = DynamicCache(config=model.config) if reuse_cache else None
    outputs = []
    for message in MESSAGES[:turns]:
        history.append({"role": "user", "content": message})
        inputs = chat_ids(history)
        if watch is not None:
            watch.next_turn(message)
        kwargs: dict[str, Any] = {"max_new_tokens": new, "min_new_tokens": new, "do_sample": False}
        if cache is not None:
            kwargs["past_key_values"] = cache
        out = model.generate(**inputs, **kwargs)
        reply = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        history.append({"role": "assistant", "content": reply})
        outputs.append(out)
    return outputs


def watch_on(kind: str, tmp_path: Path, **kwargs: Any) -> tuple[Any, Any]:
    import innards

    model = replica(kind)
    watch = innards.observe(model, tokenizer(), run="test", root=str(tmp_path), **kwargs)
    return model, watch


# --------------------------------------------------------------------- one turn per generate, light tier


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_5", "nemotron_h", "deepseek_v3"])
def test_one_record_per_generate_and_measured_equals_predicted(kind: str, tmp_path: Path) -> None:
    model, watch = watch_on(kind, tmp_path)
    try:
        converse(model, watch=watch)
        watch.wait()
        records = watch.records
        assert [r.turn for r in records] == [1, 2, 3]
        for r in records:
            assert r.runtime.backend == "observe"
            assert r.kv.measured_source == "hf_cache"
            assert r.kv.measured_bytes == r.kv.predicted_bytes
            assert r.kv.by_layer_type == dict(r.kv.predicted_by_layer_type or {})
            assert r.kv.cache_tokens == r.tokens.context_after - 1
            assert r.tokens.generated == 6
            assert r.timing.ttft_ms is not None
            assert r.timing.decode_tokens_per_s is not None
            assert r.observer.overhead_pct is not None
            assert "kv" in r.features
        assert records[1].message == MESSAGES[1]
        assert records[1].kv.growth_bytes == records[1].kv.measured_bytes - records[0].kv.measured_bytes
        assert watch.errors == 0
        assert watch.dropped_events == 0
    finally:
        watch.close()


def test_hybrid_state_in_float32_is_predicted(tmp_path: Path) -> None:
    """Gated DeltaNet keeps its recurrent state in float32 even in bfloat16 (lab finding 6)."""
    import innards

    model = replica("qwen3_5", dtype="bfloat16")
    watch = innards.observe(model, tokenizer(), root=str(tmp_path))
    try:
        converse(model, turns=1, watch=watch)
        watch.wait()
        record = watch.records[-1]
        assert record.kv.by_layer_type["recurrent"] == record.kv.predicted_by_layer_type["recurrent"]
        assert record.kv.measured_bytes == record.kv.predicted_bytes
    finally:
        watch.close()


def test_prefix_reuse_with_a_kept_cache(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3", tmp_path)
    try:
        converse(model, reuse_cache=True, watch=watch)
        watch.wait()
        first, second, third = watch.records
        assert first.tokens.cached == 0
        assert second.tokens.cached > 0
        assert second.kv.prefix_reused is True
        assert second.tokens.recomputed == second.tokens.prompt - second.tokens.cached
        assert third.kv.measured_bytes == third.kv.predicted_bytes
    finally:
        watch.close()


# --------------------------------------------------------------------- read-only: identical output


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_5", "deepseek_v4"])
def test_output_identical_with_and_without_hooks(kind: str, tmp_path: Path) -> None:
    import innards

    model = replica(kind)
    plain = [o.tolist() for o in converse(model)]
    watch = innards.observe(model, tokenizer(), root=str(tmp_path))
    try:
        observed = [o.tolist() for o in converse(model, watch=watch)]
        watch.wait()
        assert watch.records[-1].features  # the probes ran
    finally:
        watch.close()
    assert observed == plain


def test_close_removes_every_hook_and_the_wrapper(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3_5", tmp_path)
    assert "generate" in vars(model)
    watch.close()
    assert "generate" not in vars(model)
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
    converse(model, turns=1)  # still generates


def test_second_watch_on_the_same_model_is_refused(tmp_path: Path) -> None:
    import innards

    model, watch = watch_on("qwen3", tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already observed"):
            innards.observe(model)
    finally:
        watch.close()


# --------------------------------------------------------------------- features


def test_deltanet_and_gate_probes(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3_5", tmp_path)
    try:
        converse(model, watch=watch)
        watch.wait()
        record = watch.records[-1]
        assert set(record.features) == {"kv", "gated_deltanet", "gated_attention"}
        deltanet = record.features["gated_deltanet"]
        assert deltanet["data"]["layers"] == [0, 2]
        assert 0 < deltanet["summary"]["beta_mean"] < 1
        assert deltanet["data"]["state"][0]["bytes"] == 4 * 16 * 16 * 4  # heads × dk × dv, float32
        assert deltanet["summary"]["state_change_mean"] is not None  # turn 3 compares with turn 2
        gates = record.features["gated_attention"]
        assert gates["data"]["layers"] == [1, 3]
        assert len(gates["data"]["open_heads"][0]) == 4
        probes = {p.feature: p for p in record.observer.probes}
        assert probes["gated_deltanet"].hooks == 4
        assert probes["gated_attention"].hooks == 2
        assert probes["gated_deltanet"].overhead_ms is not None
        assert watch.feature_status["gated_deltanet"]["status"] == "analyzed"
        assert watch.feature_status["moe"]["status"] == "absent"
    finally:
        watch.close()


def test_latent_moe_routing(tmp_path: Path) -> None:
    model, watch = watch_on("nemotron_h", tmp_path)
    try:
        converse(model, turns=1, watch=watch)
        watch.wait()
        moe = watch.records[-1].features["moe"]
        assert moe["title"] == "LatentMoE"
        assert moe["data"]["layers"] == [1, 3]
        assert all(abs(sum(row) - 1) < 1e-3 for row in moe["data"]["load"])
        assert 0 <= moe["summary"]["balance_mean"] <= 1
        assert any("latent" in f for f in moe["findings"])
        assert watch.feature_status["mamba2"]["status"] == "not_analyzed"
    finally:
        watch.close()


def test_mhc_and_compressed_attention(tmp_path: Path) -> None:
    model, watch = watch_on("deepseek_v4", tmp_path)
    try:
        converse(model, turns=2, watch=watch)
        watch.wait()
        record = watch.records[-1]
        mhc = record.features["mhc"]
        assert mhc["summary"]["streams"] == 2
        assert mhc["summary"]["max_sum_deviation"] < 1e-4  # doubly stochastic
        assert abs(mhc["summary"]["composite_gain"] - 1) < 1e-3
        assert len(mhc["data"]["sites"]) == 8  # 4 layers × (attention, mlp)
        csa = record.features["csa_hca"]
        assert csa["summary"]["entries_match"] is True
        rows = {r["layer"]: r for r in csa["data"]["layers"]}
        tokens = record.kv.cache_tokens
        assert rows[1]["entries"] == tokens // 4
        assert rows[2]["entries"] == tokens // 128
        assert record.kv.by_layer_type["compressed"] == record.kv.predicted_by_layer_type["compressed"]
        assert record.kv.by_layer_type["indexer"] == record.kv.predicted_by_layer_type["indexer"]
        assert any("compression window fills" in f for f in record.features["kv"]["findings"])
    finally:
        watch.close()


def test_dsa_and_mla_layouts(tmp_path: Path) -> None:
    model, watch = watch_on("deepseek_v32", tmp_path)
    try:
        converse(model, turns=1, watch=watch)
        watch.wait()
        record = watch.records[-1]
        # transformers before 5.18 caches V3.2's decompressed K/V; 5.18 and later cache the latent
        expected = "expanded" if _expands_v32() else "latent"
        assert record.features["mla"]["summary"]["layout"] == expected
        assert record.features["dsa"]["summary"]["indexer_bytes"] == record.kv.by_layer_type["indexer"]
        assert record.kv.by_layer_type["indexer"] == record.kv.predicted_by_layer_type["indexer"]
    finally:
        watch.close()
    model, watch = watch_on("deepseek_v3", tmp_path)
    try:
        converse(model, turns=1, watch=watch)
        watch.wait()
        assert watch.records[-1].features["mla"]["summary"]["layout"] == "latent"
    finally:
        watch.close()


def test_explicit_analyzer_call_on_a_captured_turn(tmp_path: Path) -> None:
    from innards import MLA, GatedDeltaNet, mHC

    model, watch = watch_on("deepseek_v4", tmp_path)
    try:
        outputs = converse(model, turns=1, watch=watch)
        payload = mHC(model=model.config.to_dict(), args=watch.last_turn())
        assert payload is not None
        assert payload["summary"]["streams"] == 2
        with pytest.raises(TypeError, match="not the output of generate"):
            mHC(model="deepseek-ai/DeepSeek-V4-Flash", args=outputs[0])
        with pytest.raises(ValueError, match="was not captured"):
            GatedDeltaNet(args=watch.last_turn())
        assert MLA(args=watch.last_turn()) is None  # V4 has no MLA
    finally:
        watch.close()


def test_light_features_attach_no_hooks(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3_5", tmp_path, features="light")
    try:
        hooked = [m for m in model.modules() if m._forward_hooks]
        assert hooked == []
        converse(model, turns=1, watch=watch)
        watch.wait()
        assert set(watch.records[-1].features) == {"kv"}
        assert watch.feature_status["gated_deltanet"]["status"] == "detected"
    finally:
        watch.close()


def test_sampling_and_layer_subset(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3_5", tmp_path, sample_every=4, layers=[0])
    try:
        converse(model, turns=1, watch=watch, new=9)
        watch.wait()
        record = watch.records[-1]
        probe = next(p for p in record.observer.probes if p.feature == "gated_deltanet")
        assert probe.sampled_tokens == 3  # the prefill, then decode steps 4 and 8
        assert probe.hooks == 2  # layer 0 only
        assert record.features["gated_deltanet"]["data"]["layers"] == [0]
    finally:
        watch.close()


def test_a_failing_probe_is_fail_open(tmp_path: Path) -> None:
    from innards.features.deltanet import GatedDeltaNetAnalyzer

    original = GatedDeltaNetAnalyzer.attach

    def broken(self: Any, model: Any, detection: Any, recorder: Any, layers: Any = None) -> int:
        def boom(*_: Any) -> None:
            raise RuntimeError("probe bug")

        for name, module in model.named_modules():
            if name.endswith("in_proj_b"):
                recorder.hook(self.name, module, boom)
        return 1

    GatedDeltaNetAnalyzer.attach = broken  # type: ignore[method-assign]
    try:
        model = replica("qwen3_5")
        plain = converse(model, turns=1)[0].tolist()
        import innards

        watch = innards.observe(model, tokenizer(), root=str(tmp_path))
        observed = converse(model, turns=1, watch=watch)[0].tolist()
        watch.wait()
        probe = next(p for p in watch.records[-1].observer.probes if p.feature == "gated_deltanet")
        assert probe.errors > 0
        assert observed == plain
        watch.close()
    finally:
        GatedDeltaNetAnalyzer.attach = original  # type: ignore[method-assign]


def test_batch_and_a_user_streamer(tmp_path: Path) -> None:
    import torch
    from transformers import TextStreamer

    model, watch = watch_on("qwen3", tmp_path)
    try:
        tok = tokenizer()
        streamed: list[str] = []

        class Collect(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                streamed.append(text)

        inputs = chat_ids([{"role": "user", "content": MESSAGES[0]}])
        model.generate(**inputs, max_new_tokens=5, min_new_tokens=5, do_sample=False, streamer=Collect(tok))
        batch = {k: torch.cat([v, v]) for k, v in inputs.items()}
        model.generate(**batch, max_new_tokens=4, min_new_tokens=4, do_sample=False)
        watch.wait()
        assert streamed  # the caller's streamer still ran
        assert watch.records[-1].tokens.batch_size == 2
        assert watch.records[-1].kv.measured_bytes == watch.records[-1].kv.predicted_bytes
    finally:
        watch.close()


# --------------------------------------------------------------------- the run folder


def test_run_folder_and_new_conversation(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3", tmp_path, label="study", conversation_set="with_context")
    try:
        converse(model, turns=2, watch=watch)
        first = watch.conversation
        second = watch.new_conversation(conversation_set="basic_qa")
        converse(model, turns=1, watch=watch)
    finally:
        watch.close()
    run = tmp_path / "test"
    assert json.loads((run / "run.json").read_text())["name"] == "test"
    conv = json.loads((run / first / "conversation.json").read_text())
    assert conv["status"] == "completed"
    assert conv["turns"] == 2
    assert conv["set"] == "with_context"
    assert conv["config"]["model_type"] == "qwen3"
    assert conv["feature_status"]["kv"] == "analyzed"
    lines = (run / first / "turns.jsonl").read_text().splitlines()
    assert [json.loads(line)["turn"] for line in lines] == [1, 2]
    assert json.loads((run / second / "conversation.json").read_text())["turns"] == 1


def test_store_false_keeps_records_in_memory(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3", tmp_path, store=False)
    try:
        converse(model, turns=1, watch=watch)
        watch.wait()
        assert len(watch.records) == 1
    finally:
        watch.close()
    assert not (tmp_path / "test").exists()


def test_status(tmp_path: Path) -> None:
    model, watch = watch_on("qwen3_5", tmp_path)
    try:
        converse(model, turns=1, watch=watch)
        watch.wait()
        status = watch.status()
        assert status["turns"] == 1
        assert status["live"]["generating"] is False
        assert status["last"]["kv"]["measured_bytes"] > 0
        assert status["features"]["gated_deltanet"]["status"] == "analyzed"
        assert status["memory"]["budget_bytes"] == 16_000_000_000
        assert json.dumps(status)
    finally:
        watch.close()


def test_moe_analyzer_reads_inline_routing(tmp_path: Path) -> None:
    """observe's MoE analyzer finds blocks that route inline (MobileMoE style) through the same discovery."""
    import torch

    from innards.features import MoE
    from innards.observer.probes import ProbeRecorder
    from replicas import inline_moe_replica

    model = inline_moe_replica()
    recorder = ProbeRecorder("cpu")
    detection = MoE.detect({"num_local_experts": 6, "num_experts_per_tok": 2, "num_hidden_layers": 2})
    assert detection is not None
    assert MoE.attach(model, detection, recorder) == 2
    recorder.begin_turn()
    recorder.on_step()
    with torch.no_grad():
        model(**tokenizer()(["the cache grows with context"], return_tensors="pt"))
    captured = recorder.end_turn(None)
    counts = captured.by_feature("moe")["counts"]
    assert sorted(counts) == [0, 1]
    assert all(int(t.sum()) == 5 * 2 for rows in counts.values() for _, t in rows)  # 5 tokens, top-2
    recorder.detach()
