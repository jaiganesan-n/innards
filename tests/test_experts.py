"""Expert routing: capture on tiny replicas of each MoE family, analysis, selection and token shards.

Replicas have random weights, so only the mechanics are tested here: the hooks read exactly the
experts the model chose, never change its output, skip padding and recomputation, and the analysis
adds up. A real-checkpoint check is in test_experts_models.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from innards.experts.analysis import ExpertMap, _take
from replicas import MOE_KINDS, chat_ids, inline_moe_replica, moe_replica, replica, tokenizer

GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}


def batch(texts: list[str]) -> Any:
    tok = tokenizer()
    tok.padding_side = "left"
    return tok(texts, return_tensors="pt", padding=True)


def reference_routing(model: Any, inputs: Any) -> dict[str, Any]:
    """What each experts module receives, captured by a plain hook (independent of Innards)."""
    import torch

    seen: dict[str, Any] = {}
    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".experts") and hasattr(module, "num_experts"):

            def grab(_m: Any, args: Any, name: str = name) -> None:
                seen[name] = (args[1].detach().clone(), args[2].detach().clone())

            hooks.append(module.register_forward_pre_hook(grab))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return seen


# --------------------------------------------------------------------- capture on every family


@pytest.mark.hf
@pytest.mark.parametrize("kind", [*MOE_KINDS, "deepseek_v3", "nemotron_h"])
def test_generate_is_unchanged_and_every_token_is_counted(kind: str, tmp_path: Path) -> None:
    import torch

    import innards

    model = moe_replica(kind) if kind in MOE_KINDS else replica(kind)
    inputs = batch(["the cache grows with context", "my name is priya"])
    plain = model.generate(**inputs, max_new_tokens=4, min_new_tokens=4, **GREEDY)
    with innards.experts.trace(model, tokenizer(), run="r", root=str(tmp_path)) as trace:
        with trace.topic(["cache", "names"]), torch.no_grad():
            observed = model.generate(**inputs, max_new_tokens=4, min_new_tokens=4, **GREEDY)
        experts = trace.map()
        assert {(m.source, m.weights) for m in trace.layers} == {("experts_input", "applied")}
        assert trace.errors == 0
        assert all(m.missed == 0 for m in trace.layers)
    assert torch.equal(observed, plain)
    real = inputs["attention_mask"].sum(-1).tolist()  # left padding is never counted
    assert [experts.tokens("cache"), experts.tokens("names")] == [real[0] + 3, real[1] + 3]
    for topic in experts.topics:
        for layer, row in zip(experts.layers, experts.counts(topic), strict=True):
            top_k = experts.layer_info[experts.layers.index(layer)]["top_k"]
            assert sum(row) == experts.tokens(topic, layer) * top_k
    sequences = trace.tokens().sequences()
    assert [s["topic"] for s in sequences] == ["cache", "names"]
    assert [s["tokens"] for s in sequences] == [real[0] + 3, real[1] + 3]  # decode steps join their sequence


@pytest.mark.hf
def test_counts_equal_what_the_experts_received(tmp_path: Path) -> None:
    import torch

    import innards

    model = moe_replica("qwen3_moe")
    inputs = chat_ids([{"role": "user", "content": "tell me a story about the city at night"}])
    reference = reference_routing(model, inputs)
    trace = innards.experts.trace(model, tokenizer(), run="r", root=str(tmp_path))
    with trace.topic("story"), torch.no_grad():
        model(**inputs)
    experts = trace.map()
    routing = trace.tokens().sequence(0)
    trace.close()
    for li, moe in enumerate(trace.layers):
        index, weights = reference[moe.experts_module]
        expected = torch.bincount(index.reshape(-1), minlength=moe.experts).tolist()
        assert experts.counts("story")[li] == expected
        assert torch.equal(torch.tensor(routing["experts"])[:, li, :].long(), index)
        assert torch.equal(torch.tensor(routing["weights"])[:, li, :], weights.float())
        gate = experts.gate("story")[li]
        assert sum(g for g in gate if g is not None) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.hf
def test_inline_routing_is_read_from_the_blocks_own_topk(tmp_path: Path) -> None:
    """MobileMoE-style blocks never call their router or experts: the ``topk`` source reads them."""
    import torch

    import innards

    model = inline_moe_replica()
    inputs = batch(["the cache grows with context and", "my name is priya"])
    with torch.no_grad():
        plain = model(**inputs)
    expected = {}
    hooks = [
        layer.feed_forward.register_forward_pre_hook(
            lambda mod, args, i=i: expected.__setitem__(i, mod.routing(args[0].reshape(-1, mod.hidden_dim)))
        )
        for i, layer in enumerate(model.layers)
    ]
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    trace = innards.experts.trace(model, tokenizer(), run="r", root=str(tmp_path))
    assert [m.candidates for m in trace.layers] == [("topk",), ("topk",)]
    with trace.topic(["cache", "names"]), torch.no_grad():
        observed = model(**inputs)
    assert torch.equal(observed, plain)
    assert {(m.source, m.weights) for m in trace.layers} == {("topk", "gather")}
    routing = trace.tokens()
    valid = inputs["attention_mask"].reshape(-1).bool()
    rows = [routing.sequence(s) for s in (0, 1)]
    for li in range(2):
        index, weights = expected[li]
        got = torch.cat([torch.tensor(r["experts"])[:, li, :] for r in rows])
        got_w = torch.cat([torch.tensor(r["weights"])[:, li, :] for r in rows])
        assert torch.equal(got.long(), index[valid])
        share = got_w / got_w.sum(-1, keepdim=True)  # gathered before the model normalizes and scales
        assert torch.allclose(share, (weights / weights.sum(-1, keepdim=True))[valid].float())
    trace.close()


@pytest.mark.hf
def test_training_step_counts_each_token_once_with_checkpointing(tmp_path: Path) -> None:
    import torch

    import innards

    model = moe_replica("qwen3_moe").train()
    model.gradient_checkpointing_enable()
    inputs = batch(["the cache grows with context", "my name is priya"])
    labels = inputs["input_ids"].masked_fill(inputs["attention_mask"] == 0, -100)
    trace = innards.experts.trace(model, tokenizer(), run="r", root=str(tmp_path), tokens=False)
    with trace.topic(["cache", "names"]):
        loss = model(**inputs, labels=labels).loss
        loss.backward()  # recomputation inside backward is not a new forward
    loss2 = model(**inputs, labels=labels).loss  # outside a topic: not recorded
    loss2.backward()
    experts = trace.map()
    trace.close()
    real = inputs["attention_mask"].sum(-1).tolist()
    assert [experts.tokens("cache"), experts.tokens("names")] == real
    assert trace.forwards == 1
    assert not (trace.folder / "tokens").exists()  # tokens=False keeps the counts only
    assert torch.isfinite(loss)


@pytest.mark.hf
def test_wrong_number_of_labels_is_skipped_not_raised(tmp_path: Path) -> None:
    import torch

    import innards

    model = moe_replica("olmoe")
    inputs = batch(["the cache grows", "my name is priya"])
    trace = innards.experts.trace(model, tokenizer(), store=False)
    with trace.topic(["only-one"]), torch.no_grad():
        model(**inputs)
    assert trace.errors == 1
    assert trace.map().tokens("only-one") == 0
    with pytest.raises(TypeError):
        trace.topic(["fine", ""])
    trace.close()


@pytest.mark.hf
def test_topic_scope_restores_and_none_stops(tmp_path: Path) -> None:
    import torch

    import innards

    model = moe_replica("mixtral")
    inputs = batch(["the cache grows"])
    trace = innards.experts.trace(model, tokenizer(), store=False)
    trace.topic("outer")
    with trace.topic("inner"), torch.no_grad():
        model(**inputs)
    assert trace._label == "outer"
    with torch.no_grad():
        model(**inputs)
    trace.topic(None)
    with torch.no_grad():
        model(**inputs)
    experts = trace.map()
    assert experts.tokens("inner") == experts.tokens("outer") == 3
    assert trace.forwards == 2
    trace.close()
    assert not hasattr(model, "_innards_experts")


@pytest.mark.hf
def test_folder_layout_resume_and_merge(tmp_path: Path) -> None:
    import torch

    import innards
    from innards.experts import load, traces

    model = moe_replica("granitemoe")
    inputs = batch(["the cache grows with context"])
    for _ in range(2):  # the second trace continues the first one's folder
        trace = innards.experts.trace(model, tokenizer(), run="study", root=str(tmp_path), name="g")
        with trace, trace.topic("cache"), torch.no_grad():
            model(**inputs)
    folder = tmp_path / "study" / "experts" / "g"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["format"] == "innards.experts/1.0"
    assert summary["shards"] == ["tokens/shard-000001.safetensors", "tokens/shard-000002.safetensors"]
    assert (tmp_path / "study" / "run.json").is_file()
    assert json.loads((folder / "vocab.json").read_text())
    assert load(folder).tokens("cache") == 10
    assert [s["sequence"] for s in innards.experts.TokenRouting(folder).sequences()] == [0, 1]
    trace = innards.experts.trace(model, tokenizer(), run="study", root=str(tmp_path), name="g-rank1")
    with trace, trace.topic("names"), torch.no_grad():
        model(**inputs)
    merged = load(tmp_path / "study")
    assert merged.topics == ["cache", "names"]
    assert {t["name"] for t in traces(tmp_path)} == {"g-rank1", "g"}
    with pytest.raises(ValueError, match="holds a trace of"):
        innards.experts.trace(moe_replica("olmoe"), run="study", root=str(tmp_path), name="g")


@pytest.mark.hf
def test_token_and_expert_lookups(tmp_path: Path) -> None:
    import torch

    import innards

    model = moe_replica("qwen3_moe")
    trace = innards.experts.trace(model, tokenizer(), run="r", root=str(tmp_path))
    with trace.topic("colours"), torch.no_grad():
        model(**batch(["red green blue red green blue red"]))
    routing = trace.tokens()
    red = routing.token("red")
    assert red["occurrences"] == 3
    assert all(sum(row) == 3 * 2 for row in red["counts"])  # three tokens, top-2 each
    layer = trace.layers[0].layer
    busiest = max(range(8), key=lambda e: red["counts"][0][e])
    top = routing.expert(layer, busiest)
    assert any(row["text"] == "red" and row["routed"] == red["counts"][0][busiest] for row in top)
    assert all(0 < row["share"] <= 1 for row in top)
    trace.close()


# --------------------------------------------------------------------- analysis (no torch)


def synthetic() -> ExpertMap:
    """Two topics, one MoE layer of 4 experts, top-2: math leans on experts 0 and 1, code on 2 and 3."""
    return ExpertMap(
        {
            "model": {"id": "toy/moe", "routing": {"experts": 4, "top_k": 2}},
            "layers": [{"layer": 3, "block": "model.layers.3.mlp", "experts": 4, "top_k": 2, "weights": "applied"}],
            "topics": ["math", "code"],
            "tokens": [[100], [50]],
            "counts": [[[90, 80, 20, 10]], [[5, 5, 45, 45]]],
            "gates": [[[45.0, 35.0, 12.0, 8.0]], [[2.5, 2.5, 22.5, 22.5]]],
        }
    )


def test_selection_gate_and_lift_follow_the_definitions() -> None:
    experts = synthetic()
    assert experts.selection("math") == [[0.9, 0.8, 0.2, 0.1]]
    assert experts.gate("math") == [[0.45, 0.35, 0.12, 0.08]]
    assert experts.lift("math")[0][0] == pytest.approx(0.9 / 0.1)
    assert experts.lift("code")[0][3] == pytest.approx(0.9 / 0.1)
    assert experts.shares("math", 3) == pytest.approx([0.45, 0.4, 0.1, 0.05])
    assert [r["expert"] for r in experts.top("math", 3, n=2)] == [0, 1]
    assert [r["expert"] for r in experts.specialists("code", n=2)] == [3, 2]  # lift 9 then 4.5


def test_select_keeps_the_fewest_experts_that_reach_coverage() -> None:
    experts = synthetic()
    chosen = experts.select("math", coverage=0.8)
    assert chosen.keep(3) == [0, 1]
    assert chosen.layers[0]["covered"] == pytest.approx(0.85)
    assert experts.select("math", coverage=0.9).keep(3) == [0, 1, 2]
    assert experts.select("math", coverage=0.5, by="gate").keep(3) == [0, 1]
    assert experts.needed("code", 0.9) == [2]
    assert experts.overlap("math", "code", coverage=0.8) == [0.0]
    assert _take([0.5, 0.0, 0.5], 1.0) == ([0, 2], 1.0)  # an expert no token chose is never kept


def test_selection_export_is_labeled_derived(tmp_path: Path) -> None:
    chosen = synthetic().select("code", coverage=0.9)
    data = json.loads(chosen.save(tmp_path / "code.json").read_text())
    assert data["format"] == "innards.experts.selection/1.0"
    assert data["derived"].startswith("Derived from the measured routing of 50 'code' tokens")
    assert data["layers"][0]["keep"] == [2, 3]
    assert data["layers"][0]["block"] == "model.layers.3.mlp"
    assert data["summary"] == {"tokens": 50, "kept": 2, "of": 4, "share_kept": 0.5}
    assert "keep 2 of 4 routed experts" in str(chosen)


def test_unknown_topic_and_layer_say_what_exists() -> None:
    experts = synthetic()
    with pytest.raises(KeyError, match="measured: math, code"):
        experts.selection("poetry")
    with pytest.raises(KeyError, match=r"MoE layers: \[3\]"):
        experts.experts(7)
    with pytest.raises(ValueError, match="coverage"):
        experts.select("math", coverage=0)


def test_lift_is_none_without_other_topics() -> None:
    experts = ExpertMap({**synthetic().data, "topics": ["math"], "tokens": [[100]], "counts": [[[90, 80, 20, 10]]]})
    assert experts.lift("math") == [[None, None, None, None]]
    assert experts.specialists("math") == []
